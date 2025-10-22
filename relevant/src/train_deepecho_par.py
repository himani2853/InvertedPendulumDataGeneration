import pandas as pd
import torch
from deepecho.models import PARModel
import joblib


if __name__ == "__main__":
    # --- 1. Device setup ---
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🧠 Using device: {device}")

    # --- 2. File paths ---
    BASELINE_DATA_FILE = "../csv/stochastic_trained_agent_data_500_episodes_timesteps_50000.csv"
    MODEL_SAVE_PATH = "../models/PAR_model.pkl"
    SYNTHETIC_FILE = "../csv/synthetic_par_sequences.csv"

    # --- 3. Load baseline data ---
    df = pd.read_csv(BASELINE_DATA_FILE)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 4. Ensure timestep exists ---
    if 'timestep' not in df.columns:
        df['timestep'] = df.groupby('episode_id').cumcount()
    
    df = df.groupby('episode_id').head(200)  # keep first 200 steps per episode
    df = df.sample(frac=0.1, random_state=42)  # use 10% data

    # --- 5. Define DeepEcho schema ---
    ENTITY_COLUMN = 'episode_id'
    SEQUENCE_INDEX = 'timestep'
    CONTEXT_COLUMNS = []  # none constant across sequences

    DATA_TYPES = {
        'cart_position': 'continuous',
        'pole_angle': 'continuous',
        'cart_velocity': 'continuous',
        'pole_angular_velocity': 'continuous',
        'action': 'categorical',  # change to 'continuous' if numeric actions
        'reward': 'continuous',
        'next_cart_position': 'continuous',
        'next_pole_angle': 'continuous',
        'next_cart_velocity': 'continuous',
        'next_pole_angular_velocity': 'continuous',
        'terminated': 'categorical'
    }

    # --- 6. Initialize and train model ---
    print("🚀 Training DeepEcho PARModel...")
    model = PARModel(epochs=5, verbose=True)
    model.fit(
        data=df,
        entity_columns=[ENTITY_COLUMN],
        context_columns=CONTEXT_COLUMNS,
        data_types=DATA_TYPES,
        sequence_index=SEQUENCE_INDEX
    )
    print("✅ Model training complete.")

    # --- 7. Generate synthetic sequences ---
    print("🎲 Sampling 10 synthetic episodes...")
    synthetic_data = model.sample(num_entities=10)
    print("✅ Synthetic sample preview:")
    print(synthetic_data.head())

    # --- 8. Save results ---
    joblib.dump(model, "../models/par_model.pkl")
    print("✅ Model saved as par_model.pkl")
    # model.save(MODEL_SAVE_PATH)
    synthetic_data.to_csv(SYNTHETIC_FILE, index=False)
    print(f"📁 Synthetic data saved at: {SYNTHETIC_FILE}")
