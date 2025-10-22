import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stable_baselines3 import PPO
from analysis_utils import analyze_pendulum_data, plot_rewards_per_episode

# def analyze_pendulum_data(df, num_episodes, total_timesteps, prefix=''):
#     """
#     Loads simulation data and plots analytical graphs, using a prefix for filenames and titles.
#     """ 
#     try:
#         if not os.path.exists('plots'):
#             os.makedirs('plots')
        
#         # --- 1. Performance Analysis: Histogram of Episode Durations ---
#         episode_ends = df.index[df['terminated']].tolist()
#         last_end = -1
#         episode_lengths = []
#         for end_index in episode_ends:
#             episode_lengths.append(end_index - last_end)
#             last_end = end_index
#         if not df.empty and not df.iloc[-1]['terminated']:
#              episode_lengths.append(len(df) - last_end - 1)

#         plt.figure(figsize=(10, 6))
#         sns.histplot(episode_lengths, bins=30, kde=True)
#         plt.title(f'Histogram of Episode Durations ({prefix.capitalize()} Agent)') # CHANGED
#         plt.xlabel('Number of Steps in Episode')
#         plt.ylabel('Frequency (Number of Episodes)')
#         plt.grid(True)
        
#         filename = f'../plots/{prefix}episode_durations_histogram_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
#         plt.savefig(filename)
#         print(f"✅ Saved '{filename}'")
#         plt.close()

#         # --- 2. State-Space Analysis: Phase Plot ---
#         plt.figure(figsize=(10, 6))
#         sample_df = df.sample(n=min(len(df), 2000))
#         sns.scatterplot(data=sample_df, x='pole_angle', y='pole_angular_velocity', alpha=0.6)
#         plt.title(f'Phase Plot ({prefix.capitalize()} Agent)') # CHANGED
#         plt.xlabel('Pole Angle (radians)')
#         plt.ylabel('Pole Angular Velocity (rad/s)')
#         plt.axhline(0, color='grey', lw=0.5)
#         plt.axvline(0, color='grey', lw=0.5)
#         plt.grid(True)

#         filename = f'../plots/{prefix}phase_plot_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
#         plt.savefig(filename)
#         print(f"✅ Saved '{filename}'")
#         plt.close()

#         # --- 3. State-Space Analysis: Time Series of the Longest Episode ---
#         if episode_lengths:
#             longest_episode_duration = max(episode_lengths)
#             episode_end_indices = df.index[df['terminated']].tolist()
#             if len(episode_end_indices) < len(episode_lengths):
#                 episode_end_indices.append(len(df) - 1)
#             longest_episode_end_index = episode_end_indices[episode_lengths.index(longest_episode_duration)]
#             longest_episode_start_index = longest_episode_end_index - longest_episode_duration + 1
#             longest_episode_df = df.iloc[longest_episode_start_index:longest_episode_end_index+1].reset_index(drop=True)
            
#             plt.figure(figsize=(14, 8))
#             plt.plot(longest_episode_df.index, longest_episode_df['pole_angle'], label='Pole Angle')
#             plt.plot(longest_episode_df.index, longest_episode_df['cart_position'], label='Cart Position')
#             plt.plot(longest_episode_df.index, longest_episode_df['action'], label='Action', linestyle='--')
#             plt.title(f'Time Series of Longest Episode ({prefix.capitalize()} Agent)') # CHANGED
#             plt.xlabel('Step Number')
#             plt.ylabel('Value')
#             plt.legend()
#             plt.grid(True)

#             filename = f'../plots/{prefix}longest_episode_timeseries_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
#             plt.savefig(filename)
#             print(f"✅ Saved '{filename}'")
#             plt.close()

#     except FileNotFoundError:
#         print(f"Error: Could not find data to analyze.")
#         return

# def plot_rewards_per_episode(df, num_episodes, total_timesteps, prefix=''):
#     """
#     Loads simulation data and plots the total reward for each episode, using a prefix.
#     """
#     try:
#         if not os.path.exists('plots'):
#             os.makedirs('plots')

#         # --- Calculate Total Reward for Each Episode ---
#         episode_ends = df.index[df['terminated']].tolist()
#         last_end = -1
#         episode_rewards = []
#         for end_index in episode_ends:
#             episode_reward = df.iloc[last_end + 1:end_index + 1]['reward'].sum()
#             episode_rewards.append(episode_reward)
#             last_end = end_index
#         if not df.empty and not df.iloc[-1]['terminated']:
#             last_episode_reward = df.iloc[last_end + 1:]['reward'].sum()
#             episode_rewards.append(last_episode_reward)

#         # --- Plotting ---
#         plt.figure(figsize=(12, 7))
#         episode_numbers = range(1, len(episode_rewards) + 1)
#         sns.barplot(x=list(episode_numbers), y=episode_rewards, palette="viridis")
#         plt.title(f'Total Reward per Episode ({prefix.capitalize()} Agent)') # CHANGED
#         plt.xlabel('Episode Number')
#         plt.ylabel('Total Reward (Number of Steps)')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
        
#         average_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
#         plt.axhline(average_reward, color='r', linestyle='--', label=f'Average Reward: {average_reward:.2f}')
#         plt.legend()
        
#         filename = f'../plots/{prefix}rewards_per_episode_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
#         plt.savefig(filename)
#         print(f"✅ Saved '{filename}'")
#         plt.close()

#     except FileNotFoundError:
#         print(f"Error: Could not find data to analyze.")
#         return


# --- STEP 1: A NEW FUNCTION TO TRAIN THE AGENT ---
def train_agent(env):
    """
    Trains a PPO agent and returns the trained model.
    """
    print("Starting agent training...")
    # "MlpPolicy" means the agent uses a neural network policy
    model = PPO("MlpPolicy", env, verbose=1)            
    # The learning happens in this single line
    model.learn(total_timesteps=20000)
    print("✅ Training complete!")
    return model


# --- STEP 2: MODIFIED FUNCTION TO GENERATE DATA WITH THE TRAINED AGENT ---
def generate_data_with_trained_agent(env, model, num_episodes=50):
    """
    Runs simulations using a trained model to generate performance data.
    """
    collected_data = []
    print(f"Generating data with trained agent for {num_episodes} episodes...")

    for i in range(num_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # *** KEY CHANGE HERE ***
            # We ask the trained model for the best action instead of a random one.
            # action, _ = model.predict(observation, deterministic=True)
            action, _ = model.predict(observation, deterministic=False)  # Use stochastic actions for more varied data
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # The action from model.predict is a numpy array, so we use it directly
            collected_data.append([
                i,
                observation[0], observation[1], observation[2], observation[3],
                action[0], reward,
                next_observation[0], next_observation[1], next_observation[2], next_observation[3],
                terminated
            ])
            observation = next_observation

    # Create a Pandas DataFrame for easier analysis
    columns = [
        'episode_id',
        'cart_position', 'pole_angle', 'cart_velocity', 'pole_angular_velocity',
        'action', 'reward',
        'next_cart_position', 'next_pole_angle', 'next_cart_velocity', 'next_pole_angular_velocity',
        'terminated'
    ]
    df = pd.DataFrame(collected_data, columns=columns)
    return df


# --- STEP 3: NEW MAIN EXECUTION WORKFLOW ---
if __name__ == "__main__":
    # Create the environment once
    env = gym.make("InvertedPendulum-v5")
    
    # 1. Train the agent
    total_timesteps = 50000  # Set this to match the value used in model.learn
    trained_model = PPO("MlpPolicy", env, verbose=1)
    trained_model.learn(total_timesteps=total_timesteps)
    print("✅ Training complete!")
    
    # 2. Generate data using the smart, trained agent
    # num_episodes_to_evaluate = 50
    num_episodes_to_evaluate = 500  # More episodes for better data
    trained_agent_data = generate_data_with_trained_agent(env, trained_model, num_episodes_to_evaluate)
    
    # 3. Save and analyze the new data
    filepath = f"../csv/stochastic_trained_agent_data_{num_episodes_to_evaluate}_episodes_timesteps_{total_timesteps}.csv"
    trained_agent_data.to_csv(filepath, index=False)
    print(f"\n✅ Trained agent data saved to {filepath}")
    
    # Use your existing plotting functions on the new, high-quality data
    analyze_pendulum_data(trained_agent_data, num_episodes_to_evaluate, total_timesteps, prefix='stochastic_trained_')
    plot_rewards_per_episode(trained_agent_data, num_episodes_to_evaluate, total_timesteps, prefix='stochastic_trained_')
    print("\n✅ Analysis of trained agent is complete. Check the 'plots' folder.")

    env.close()