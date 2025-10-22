import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

# --- 1. Configuration ---
HYBRID_DATA_DIR = "../csv/hybrid_datasets/"
MODEL_SAVE_DIR = "../models/bc_agents/"

# --- Define the ratios to loop through ---
MIX_RATIOS = [0, 10, 25, 50, 75, 90, 100]

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# --- 2. Define State and Action Columns ---
# These are your inputs (X)
STATE_COLS = [
    'cart_position', 
    'pole_angle', 
    'cart_velocity', 
    'pole_angular_velocity'
]
# This is your output (y)
ACTION_COL = ['action']

# --- Start the main training loop ---
for ratio in MIX_RATIOS:
    print("\n" + "="*50)
    print(f"🚀 STARTING TRAINING FOR: {ratio}% REAL DATA MIX")
    print("="*50 + "\n")

    # --- 3. Generate file paths dynamically ---
    MODEL_NAME = f"hybrid_mix_{ratio}_real"
    HYBRID_DATA_FILE = os.path.join(HYBRID_DATA_DIR, f"{MODEL_NAME}.csv")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_model.h5")

    if not os.path.exists(HYBRID_DATA_FILE):
        print(f"⚠️ Warning: File not found, skipping. {HYBRID_DATA_FILE}")
        continue # Skip to the next loop iteration

    # --- 4. Load and Prepare Data ---
    print(f"Loading data from {HYBRID_DATA_FILE}...")
    df = pd.read_csv(HYBRID_DATA_FILE)

    # Separate inputs (X) and outputs (y)
    X = df[STATE_COLS]
    y = df[ACTION_COL]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples.")

    # --- 5. Build the Neural Network Model ---
    # We re-initialize the model from scratch for each loop
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(STATE_COLS)]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1) 
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Only print summary for the first model
    if ratio == MIX_RATIOS[0]:
        model.summary()

    # --- 6. Train the Model ---
    print(f"\nStarting model training for {MODEL_NAME}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        verbose=2 
    )

    print("✅ Training complete.")

    # --- 7. Save the Trained Agent ---
    model.save(MODEL_SAVE_PATH)
    print(f"🎉 Agent saved to: {MODEL_SAVE_PATH}")

print("\n" + "="*50)
print("🎉 ALL AGENTS TRAINED AND SAVED. 🎉")
print("="*50)