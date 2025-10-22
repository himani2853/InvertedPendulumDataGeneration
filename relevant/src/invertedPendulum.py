import gymnasium as gym
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_inverted_pendulum_data(num_episodes=1):
    """
    Runs simulations in the InvertedPendulum-v5 environment to generate data.

    Args:
        num_episodes (int): The number of episodes to simulate.

    Returns:
        pandas.DataFrame: A DataFrame containing the collected state transitions.
    """
    # 1. Create the Inverted Pendulum environment
    env = gym.make("InvertedPendulum-v5")

    # A list to store all the data we collect
    collected_data = []

    print(f"🚀 Starting data generation for {num_episodes} episodes...")

    for i in range(num_episodes):
        # 2. Reset the environment for a new episode
        # This returns the initial observation (state)
        observation, info = env.reset()
        print("observation", observation)
        terminated = False
        truncated = False
        episode_reward = 0

        # 3. Loop until the episode ends
        while not terminated and not truncated:
            
            # 4. Choose a random action
            # The action space is Box(-3.0, 3.0), so we sample from it.
            action = env.action_space.sample()
            print("Action: ", action)
            # 5. Take the action and get the result from the environment
            # This returns the new state, the reward, and done flags.
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Store the experience tuple
            # (current_state, action, reward, next_state, is_terminal)
            collected_data.append([
                observation[0], # cart_position
                observation[1], # pole_angle
                observation[2], # cart_velocity
                observation[3], # pole_angular_velocity
                action[0],      # action is a 1-element array
                reward,
                next_observation[0],
                next_observation[1],
                next_observation[2],
                next_observation[3],
                terminated
            ])
            
            # 6. Update the current observation for the next loop iteration
            observation = next_observation
            episode_reward += reward
            print("Terminated: ", terminated, "Truncated: ", truncated)

        print(f"Episode {i+1}: Total Reward = {episode_reward}")

    # 7. Close the environment
    env.close()
    
    # Create a Pandas DataFrame for easier analysis
    columns = [
        'cart_position', 'pole_angle', 'cart_velocity', 'pole_angular_velocity',
        'action', 'reward',
        'next_cart_position', 'next_pole_angle', 'next_cart_velocity', 'next_pole_angular_velocity',
        'terminated'
    ]
    df = pd.DataFrame(collected_data, columns=columns)
    
    return df

def analyze_pendulum_data(num_episodes, filepath):
    """
    Loads the simulation data and plots several analytical graphs.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please run the data generation script first.")
        return

    # --- 1. Performance Analysis: Histogram of Episode Durations ---
    # To get episode lengths, we find where episodes end (terminated=True)
    # and calculate the number of steps between each end.
    episode_ends = df.index[df['terminated']].tolist()
    last_end = -1
    episode_lengths = []
    for end_index in episode_ends:
        episode_lengths.append(end_index - last_end)
        last_end = end_index

    plt.figure(figsize=(10, 6))
    sns.histplot(episode_lengths, bins=30, kde=True)
    plt.title('Histogram of Episode Durations (Performance)')
    plt.xlabel('Number of Steps in Episode')
    plt.ylabel('Frequency (Number of Episodes)')
    plt.grid(True)
    plt.savefig(f'../plots/episode_durations_histogram_{num_episodes}.png')
    print(f"✅ Saved 'episode_durations_histogram_{num_episodes}.png'")


    # --- 2. State-Space Analysis: Phase Plot ---
    plt.figure(figsize=(10, 6))
    # We use a subset of the data to avoid overplotting
    sample_df = df.sample(n=min(len(df), 2000))
    sns.scatterplot(data=sample_df, x='pole_angle', y='pole_angular_velocity', alpha=0.6)
    plt.title('Phase Plot: Pole Angle vs. Pole Angular Velocity')
    plt.xlabel('Pole Angle (radians)')
    plt.ylabel('Pole Angular Velocity (rad/s)')
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid(True)
    plt.savefig(f'./plots/phase_plot_{num_episodes}.png')
    print(f"✅ Saved 'phase_plot_{num_episodes}.png'")


    # --- 3. State-Space Analysis: Time Series of the Longest Episode ---
    if episode_lengths:
        longest_episode_duration = max(episode_lengths)
        longest_episode_end_index = episode_ends[episode_lengths.index(longest_episode_duration)]
        longest_episode_start_index = longest_episode_end_index - longest_episode_duration + 1
        
        longest_episode_df = df.iloc[longest_episode_start_index:longest_episode_end_index+1].reset_index(drop=True)
        
        plt.figure(figsize=(14, 8))
        plt.plot(longest_episode_df.index, longest_episode_df['pole_angle'], label='Pole Angle')
        plt.plot(longest_episode_df.index, longest_episode_df['cart_position'], label='Cart Position')
        plt.plot(longest_episode_df.index, longest_episode_df['action'], label='Action', linestyle='--')
        plt.title(f'Time Series of the Longest Episode ({longest_episode_duration} steps)')
        plt.xlabel('Step Number')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../plots/longest_episode_timeseries_{num_episodes}.png')
        print(f"✅ Saved 'longest_episode_timeseries_{num_episodes}.png'")

def plot_rewards_per_episode(num_episodes, filepath):
    """
    Loads simulation data and plots the total reward for each episode.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure you have run the data generation script first.")
        return

    # --- Calculate Total Reward for Each Episode ---
    episode_ends = df.index[df['terminated']].tolist()
    last_end = -1
    episode_rewards = []
    for end_index in episode_ends:
        # Sum the 'reward' column for the slice of the DataFrame representing one episode
        episode_reward = df.iloc[last_end + 1:end_index + 1]['reward'].sum()
        episode_rewards.append(episode_reward)
        last_end = end_index

    # Handle the last episode if it was truncated and not terminated
    if not df.empty and not df.iloc[-1]['terminated']:
        last_episode_reward = df.iloc[last_end + 1:]['reward'].sum()
        episode_rewards.append(last_episode_reward)

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    episode_numbers = range(1, len(episode_rewards) + 1)
    
    sns.barplot(x=list(episode_numbers), y=episode_rewards, palette="viridis")
    
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward (Number of Steps)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a line for the average reward
    average_reward = sum(episode_rewards) / len(episode_rewards)
    plt.axhline(average_reward, color='r', linestyle='--', label=f'Average Reward: {average_reward:.2f}')
    plt.legend()

    plt.savefig(f'../plots/rewards_per_episode_{num_episodes}.png')
    print(f"✅ Saved '../plots/rewards_per_episode_{num_episodes}.png'")


# --- Main execution ---
if __name__ == "__main__":
    # Generate data from 5 episodes
    num_episodes_to_run = (10, 50, 100)
    
    for num_episodes in num_episodes_to_run:
        filepath = f"../csv/inverted_pendulum_data_{num_episodes}.csv"
        simulation_data = generate_inverted_pendulum_data(num_episodes)
        # print("\n--- Generated Data (first 5 rows) ---")
        # print(simulation_data.head())
        simulation_data.to_csv(filepath, index=False)
        print(f"\n✅ Data saved to {filepath}")
        analyze_pendulum_data(num_episodes, filepath)
        plot_rewards_per_episode(num_episodes, filepath)