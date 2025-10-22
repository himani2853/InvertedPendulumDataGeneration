import pandas as pd
from ctgan import CTGAN
import os

# Import your plotting functions from the new utility file
from analysis_utils import analyze_pendulum_data, plot_rewards_per_episode

# --- Configuration ---
EPOCHS = 100
BASELINE_DATA_FILE = "../csv/stochastic_trained_agent_data_500_episodes_timesteps_50000.csv"
MODEL_PATH = f"../models/ctgan_model_{EPOCHS}.pkl"
NUM_EPISODES = 500  # Match the original data collection
TOTAL_TIMESTEPS = 50000 # Match the original data collection
SYNTHETIC_DATA_FILE = f"../csv/ctgan_synthetic_data_{NUM_EPISODES}_episodes_{TOTAL_TIMESTEPS}_timesteps.csv"

def main():
    # --- 1. Load the Trained Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please run the training script first.")
        return
        
    print(f"Loading trained model from '{MODEL_PATH}'...")
    model = CTGAN.load(MODEL_PATH)

    # --- 2. Load Original Baseline Data ---
    if not os.path.exists(BASELINE_DATA_FILE):
        print(f"Error: Baseline data file not found at '{BASELINE_DATA_FILE}'.")
        return
        
    baseline_data = pd.read_csv(BASELINE_DATA_FILE)
    print("Baseline data loaded.")

    # --- 3. Generate Synthetic Data ---
    print(f"Generating {len(baseline_data)} rows of synthetic data...")
    synthetic_data = model.sample(len(baseline_data))
    print("Synthetic data generation complete.")
    synthetic_data.to_csv(SYNTHETIC_DATA_FILE, index=False)
    print(f"✅ Synthetic data generated and saved to '{SYNTHETIC_DATA_FILE}'")
    print("----------------------------------------")

    # --- 4. Analyze Both Datasets for Comparison ---
    print("📊 Analyzing BASELINE data...")
    analyze_pendulum_data(baseline_data, NUM_EPISODES, TOTAL_TIMESTEPS, prefix='baseline_')
    plot_rewards_per_episode(baseline_data, NUM_EPISODES, TOTAL_TIMESTEPS, prefix='baseline_')
    print("✅ Baseline analysis complete.")
    print("----------------------------------------")

    print("📊 Analyzing SYNTHETIC data...")
    analyze_pendulum_data(synthetic_data, NUM_EPISODES, TOTAL_TIMESTEPS, prefix=f'synthetic_{EPOCHS}')
    plot_rewards_per_episode(synthetic_data, NUM_EPISODES, TOTAL_TIMESTEPS, prefix=f'synthetic_{EPOCHS}')
    print("✅ Synthetic analysis complete.")
    print("----------------------------------------")
    
    print("🎉 All done! Compare the plots in the 'plots/' directory.")

if __name__ == "__main__":
    main()