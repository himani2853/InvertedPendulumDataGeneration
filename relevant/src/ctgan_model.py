import pandas as pd
# import sdv
import torch
# from ctgan import CTGAN
from sdv.sequential import PARModel  # <-- 1. Import PARModel

# Check for Mac's GPU
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ MPS backend is available. Using Mac's GPU.")
else:
    device = "cpu"
    print("⚠️ MPS backend not available. Falling back to CPU.")

# The CSV file you created with the diverse, stochastic agent data
BASELINE_DATA_FILE = "../csv/stochastic_trained_agent_data_500_episodes_timesteps_50000.csv"
MODEL_SAVE_PATH = "../models/ctgan_model.pkl"

# --- 2. Define the new 'episode_id' column ---
EPISODE_ID_COLUMN = 'episode_id'

# Load your baseline expert data
print(f"Loading baseline data from {BASELINE_DATA_FILE}...")
baseline_data = pd.read_csv(BASELINE_DATA_FILE)

# --- 3. Initialize PARModel instead of CTGAN ---
# We tell it which column identifies the sequences
model = PARModel(
    sequence_key=EPISODE_ID_COLUMN,
    epochs=50,  # Start with 50-100 epochs. 1 is not enough.
    verbose=True,
    cuda=device # Pass the device ("mps" or "cpu")
)

print("Training PARModel...")
model.fit(baseline_data)
print("PARModel fitting complete.")

# Initialize and train the CTGAN model
# You can adjust epochs for longer/shorter training
# model = CTGAN(epochs=1, verbose=True)
# print("Training CTGAN model...")
# model.fit(baseline_data)

# Save the trained model for later use
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete and saved to '{MODEL_SAVE_PATH}'")