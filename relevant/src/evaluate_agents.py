import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd

# --- 1. Configuration ---
MODELS_DIR = "../models/bc_agents/"
NUM_EPISODES_TO_RUN = 100

# Define the ratios that were trained
MIX_RATIOS = [0, 10, 25, 50, 75, 90, 100]

# A list to store all our final results
all_results = []

# --- 2. Create the environment ---
env = gym.make("InvertedPendulum-v5")

# --- 3. Main Evaluation Loop ---
for ratio in MIX_RATIOS:
    print("\n" + "="*50)
    print(f"🔬 EVALUATING AGENT: {ratio}% REAL DATA MIX")
    print("="*50 + "\n")

    # --- 4. Load the trained agent ---
    MODEL_NAME = f"hybrid_mix_{ratio}_real_model.h5"
    MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Model file not found, skipping. {MODEL_PATH}")
        continue

    try:
        # Try loading normally first (this will attempt to restore optimizer/metrics)
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Loaded agent: {MODEL_NAME}")
    except Exception as e:
        # Some models saved in legacy HDF5 format include training config objects
        # (loss/metrics) that may not deserialize cleanly across Keras/TensorFlow
        # versions. For evaluation/inference we don't need the training config,
        # so load without compiling as a robust fallback.
        print(f"⚠️ Warning: failed to fully load model (will retry with compile=False): {e}")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Loaded agent (compile=False): {MODEL_NAME}")

    episode_rewards = []
    print(f"Running evaluation for {NUM_EPISODES_TO_RUN} episodes...")

    for i in range(NUM_EPISODES_TO_RUN):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            # --- 5. Get Action from Your Agent ---
            # Reshape the state to (1, 4) because the model expects a batch
            state_reshaped = np.array(state).reshape(1, -1)
            
            # Get action from the model
            action = model.predict(state_reshaped, verbose=0)[0]
            
            # --- 6. Take the action in the environment ---
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            state = next_state
        
        # Episode finished
        episode_rewards.append(total_reward)
    
    # --- 7. Calculate and Print Agent's Results ---
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"✅ Evaluation complete for {ratio}% agent.")
    print(f"   Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Store the result
    all_results.append({
        'percent_real': ratio,
        'mean_reward': mean_reward,
        'std_dev': std_reward
    })

env.close()

# --- 8. Print Final Summary Table ---
print("\n" + "="*60)
print("🎉 FINAL EVALUATION RESULTS 🎉")
print("="*60 + "\n")

# Create a DataFrame for easy viewing
results_df = pd.DataFrame(all_results)
results_df = results_df.set_index('percent_real')
print(results_df.to_markdown(floatfmt=".2f"))

# Save results to a CSV file for plotting
results_df.to_csv("../csv/final_evaluation_results.csv")
print("\n📁 Results saved to ../csv/final_evaluation_results.csv")
print("You can now use this CSV to create your final plot.")