import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_pendulum_data(df, num_episodes, total_timesteps, prefix=''):
    """
    Loads simulation data and plots analytical graphs, using a prefix for filenames and titles.
    """ 
    try:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # --- 1. Performance Analysis: Histogram of Episode Durations ---
        episode_ends = df.index[df['terminated']].tolist()
        last_end = -1
        episode_lengths = []
        for end_index in episode_ends:
            episode_lengths.append(end_index - last_end)
            last_end = end_index
        if not df.empty and not df.iloc[-1]['terminated']:
             episode_lengths.append(len(df) - last_end - 1)

        plt.figure(figsize=(10, 6))
        sns.histplot(episode_lengths, bins=30, kde=True)
        plt.title(f'Histogram of Episode Durations ({prefix.capitalize()} Agent)') # CHANGED
        plt.xlabel('Number of Steps in Episode')
        plt.ylabel('Frequency (Number of Episodes)')
        plt.grid(True)
        
        filename = f'../plots/{prefix}episode_durations_histogram_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
        plt.savefig(filename)
        print(f"✅ Saved '{filename}'")
        plt.close()

        # --- 2. State-Space Analysis: Phase Plot ---
        plt.figure(figsize=(10, 6))
        sample_df = df.sample(n=min(len(df), 2000))
        sns.scatterplot(data=sample_df, x='pole_angle', y='pole_angular_velocity', alpha=0.6)
        plt.title(f'Phase Plot ({prefix.capitalize()} Agent)') # CHANGED
        plt.xlabel('Pole Angle (radians)')
        plt.ylabel('Pole Angular Velocity (rad/s)')
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.grid(True)

        filename = f'../plots/{prefix}phase_plot_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
        plt.savefig(filename)
        print(f"✅ Saved '{filename}'")
        plt.close()

        # --- 3. State-Space Analysis: Time Series of the Longest Episode ---
        if episode_lengths:
            longest_episode_duration = max(episode_lengths)
            episode_end_indices = df.index[df['terminated']].tolist()
            if len(episode_end_indices) < len(episode_lengths):
                episode_end_indices.append(len(df) - 1)
            longest_episode_end_index = episode_end_indices[episode_lengths.index(longest_episode_duration)]
            longest_episode_start_index = longest_episode_end_index - longest_episode_duration + 1
            longest_episode_df = df.iloc[longest_episode_start_index:longest_episode_end_index+1].reset_index(drop=True)
            
            plt.figure(figsize=(14, 8))
            plt.plot(longest_episode_df.index, longest_episode_df['pole_angle'], label='Pole Angle')
            plt.plot(longest_episode_df.index, longest_episode_df['cart_position'], label='Cart Position')
            plt.plot(longest_episode_df.index, longest_episode_df['action'], label='Action', linestyle='--')
            plt.title(f'Time Series of Longest Episode ({prefix.capitalize()} Agent)') # CHANGED
            plt.xlabel('Step Number')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            filename = f'../plots/{prefix}longest_episode_timeseries_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
            plt.savefig(filename)
            print(f"✅ Saved '{filename}'")
            plt.close()

    except FileNotFoundError:
        print(f"Error: Could not find data to analyze.")
        return

def plot_rewards_per_episode(df, num_episodes, total_timesteps, prefix=''):
    """
    Loads simulation data and plots the total reward for each episode, using a prefix.
    """
    try:
        if not os.path.exists('plots'):
            os.makedirs('plots')

        # --- Calculate Total Reward for Each Episode ---
        episode_ends = df.index[df['terminated']].tolist()
        last_end = -1
        episode_rewards = []
        for end_index in episode_ends:
            episode_reward = df.iloc[last_end + 1:end_index + 1]['reward'].sum()
            episode_rewards.append(episode_reward)
            last_end = end_index
        if not df.empty and not df.iloc[-1]['terminated']:
            last_episode_reward = df.iloc[last_end + 1:]['reward'].sum()
            episode_rewards.append(last_episode_reward)

        # --- Plotting ---
        plt.figure(figsize=(12, 7))
        episode_numbers = range(1, len(episode_rewards) + 1)
        sns.barplot(x=list(episode_numbers), y=episode_rewards, palette="viridis", hue=list(episode_numbers))
        plt.title(f'Total Reward per Episode ({prefix.capitalize()} Agent)') # CHANGED
        plt.xlabel('Episode Number')
        plt.ylabel('Total Reward (Number of Steps)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        average_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        plt.axhline(average_reward, color='r', linestyle='--', label=f'Average Reward: {average_reward:.2f}')
        plt.legend()
        
        filename = f'../plots/{prefix}rewards_per_episode_{num_episodes}_timesteps_{total_timesteps}.png' # CHANGED
        plt.savefig(filename)
        print(f"✅ Saved '{filename}'")
        plt.close()

    except FileNotFoundError:
        print(f"Error: Could not find data to analyze.")
        return

