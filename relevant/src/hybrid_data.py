import pandas as pd
from sklearn.utils import shuffle
import os

# --- Configuration ---
REAL_DATA_FILE = "../csv/stochastic_trained_agent_data_500_episodes_timesteps_50000.csv"
SYNTHETIC_DATA_FILE = "../csv/synthetic_par_sequences.csv" # Use your PAR model's output
HYBRID_DATA_DIR = "../csv/hybrid_datasets/"

# Define the ratios of REAL data you want in your mix
# 0.0 = 100% Synthetic
# 0.5 = 50% Real, 50% Synthetic
# 1.0 = 100% Real
MIX_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# Create the output directory if it doesn't exist
if not os.path.exists(HYBRID_DATA_DIR):
    os.makedirs(HYBRID_DATA_DIR)
    print(f"Created directory: {HYBRID_DATA_DIR}")

# --- Load Data ---
print(f"Loading REAL data from {REAL_DATA_FILE}...")
real_data = pd.read_csv(REAL_DATA_FILE)
print(f"Loading SYNTHETIC data from {SYNTHETIC_DATA_FILE}...")
synthetic_data = pd.read_csv(SYNTHETIC_DATA_FILE)

# Use the size of the *real* dataset as the target size for all hybrid sets
TOTAL_SIZE = len(real_data)
print(f"Target size for all hybrid datasets will be {TOTAL_SIZE} rows.")

# --- Create Hybrid Datasets ---
for ratio in MIX_RATIOS:
    print(f"--- Creating mix for {ratio*100}% REAL data ---")
    
    # 1. Calculate number of samples for each
    n_real = int(TOTAL_SIZE * ratio)
    n_synthetic = TOTAL_SIZE - n_real
    
    # 2. Sample from each dataframe
    # We use .sample() to get a random subset
    real_sample = real_data.sample(n=n_real, replace=True)
    
    # Handle the 0% real / 100% synthetic case
    if n_synthetic > 0:
        synthetic_sample = synthetic_data.sample(n=n_synthetic, replace=True)
        # 3. Combine them
        hybrid_df = pd.concat([real_sample, synthetic_sample])
    else:
        hybrid_df = real_sample

    # 4. Shuffle the combined dataset
    # This is CRITICAL so the agent doesn't learn from all 
    # the real data at once.
    hybrid_df = shuffle(hybrid_df)
    
    # 5. Save the new dataset
    output_filename = f"hybrid_mix_{int(ratio*100)}_real.csv"
    output_path = os.path.join(HYBRID_DATA_DIR, output_filename)
    hybrid_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {output_filename} ({n_real} real, {n_synthetic} synthetic)")

print("\n🎉 All hybrid datasets created!")