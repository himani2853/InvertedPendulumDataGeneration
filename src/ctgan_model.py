import pandas as pd
# import sdv
from ctgan import CTGAN

# The CSV file you created with the diverse, stochastic agent data
BASELINE_DATA_FILE = "../csv/stochastic_trained_agent_data_500_episodes_timesteps_50000.csv"
EPOCHS = 100
MODEL_SAVE_PATH = f"../models/ctgan_model_{EPOCHS}.pkl"

# Load your baseline expert data
print(f"Loading baseline data from {BASELINE_DATA_FILE}...")
baseline_data = pd.read_csv(BASELINE_DATA_FILE)

# Initialize and train the CTGAN model
# You can adjust epochs for longer/shorter training
device = "mps"
model = CTGAN(epochs=EPOCHS, verbose=True, cuda=device)
print("Loaded CTGAN model...")
model.fit(baseline_data)
print("CTGAN fitting complete.")

# Save the trained model for later use
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete and saved to '{MODEL_SAVE_PATH}'")