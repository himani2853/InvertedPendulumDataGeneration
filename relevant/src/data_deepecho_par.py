import os
import pandas as pd
import joblib  # for loading the saved PARModel

# Import analysis utilities
from analysis_utils import analyze_pendulum_data, plot_rewards_per_episode

# --- Configuration ---
BASELINE_DATA_FILE = "../csv/stochastic_trained_agent_data_500_episodes_timesteps_50000.csv"
MODEL_PATH = "../models/par_model.pkl"
SYNTHETIC_DATA_FILE = "../csv/par_synthetic_data_500_episodes_timesteps_50000.csv"

# These match the experimental setup
NUM_EPISODES = 500
TOTAL_TIMESTEPS = 50000

# The data is sampled:
MAX_STEPS_PER_EPISODE = 200
DATA_FRACTION = 0.1


def main():
    # --- 1. Load the Trained Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'. Please train the PARModel first.")
        return

    print(f"📦 Loading trained PARModel from '{MODEL_PATH}'...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # --- 2. Load Original Baseline Data ---
    if not os.path.exists(BASELINE_DATA_FILE):
        print(f"❌ Error: Baseline data file not found at '{BASELINE_DATA_FILE}'.")
        return

    # baseline_data = pd.read_csv(BASELINE_DATA_FILE)
    df = pd.read_csv(BASELINE_DATA_FILE)

    print(f"Original dataset shape: {df.shape}")
    # Keep first 200 timesteps per episode
    # df = df.groupby("episode_id").head(MAX_STEPS_PER_EPISODE)
    # # Use only 10% of the data for training/testing
    # df = df.sample(frac=DATA_FRACTION, random_state=42).reset_index(drop=True)
    print(f"✅ Reduced dataset shape: {df.shape}")

    # print(f"✅ Baseline data loaded: {baseline_data.shape[0]} rows, {baseline_data.shape[1]} columns")

    # --- 3. Generate Synthetic Data ---
    print(f"🚀 Generating synthetic data for {NUM_EPISODES} episodes...")

    # `num_entities` = number of episode sequences to sample
    synthetic_data = model.sample(num_entities=NUM_EPISODES)

    # --- 4. Save the Synthetic Data ---
    synthetic_data.to_csv(SYNTHETIC_DATA_FILE, index=False)
    print(f"✅ Synthetic data generated and saved to '{SYNTHETIC_DATA_FILE}'")
    print("----------------------------------------")

    # --- 5. Comparative Analysis ---
    print("📊 Analyzing BASELINE data...")
    analyze_pendulum_data(df, NUM_EPISODES, TOTAL_TIMESTEPS, prefix="baseline_")
    plot_rewards_per_episode(df, NUM_EPISODES, TOTAL_TIMESTEPS, prefix="baseline_")
    print("✅ Baseline analysis complete.")
    print("----------------------------------------")

    print("📊 Analyzing SYNTHETIC data...")
    analyze_pendulum_data(synthetic_data, NUM_EPISODES, TOTAL_TIMESTEPS, prefix="synthetic_")
    plot_rewards_per_episode(synthetic_data, NUM_EPISODES, TOTAL_TIMESTEPS, prefix="synthetic_")
    print("✅ Synthetic analysis complete.")
    print("----------------------------------------")

    print("🎉 All done! Compare the plots in the 'plots/' directory.")


if __name__ == "__main__":
    main()
