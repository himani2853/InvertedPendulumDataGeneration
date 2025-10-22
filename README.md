# InvertedPendulumDataGeneration

BTP - Project 2

This repository contains code for generating data from the OpenAI Gym InvertedPendulum environment, training a PPO agent, and producing/analysing synthetic data with CTGAN. Below is a short description of the main scripts under `src/` and cross-file notes to help you maintain and run the project.

## Files and purpose

- `analysis_utils.py`
	- Purpose: plotting / analysis utilities reused by other scripts.
	- Key functions:
		- `analyze_pendulum_data(df, num_episodes, total_timesteps, prefix='')` — histogram, phase plot, timeseries of longest episode; saves PNGs.
		- `plot_rewards_per_episode(df, num_episodes, total_timesteps, prefix='')` — bar plot of total reward per episode; saves PNG.
	- Inputs/outputs: expects a pandas DataFrame with columns like `pole_angle`, `pole_angular_velocity`, `reward`, `terminated`. Writes images to `../plots` with filenames containing `num_episodes` and `total_timesteps`.
	- Notes: path handling and directory creation are present but not fully consistent; consider standardizing output directories.

- `ctgan_model.py`
	- Purpose: train a CTGAN on baseline CSV and save the model.
	- Behavior: loads CSV `BASELINE_DATA_FILE`, constructs a CTGAN, fits and saves the model to `../models/ctgan_model.pkl`.
	- Dependencies: `pandas` and a CTGAN implementation (`sdv` or `ctgan`).
	- Notes: there are two CTGAN APIs used across scripts; pick `sdv` or `ctgan` and align the code.

- `ctgan_data.py`
	- Purpose: load a trained CTGAN model, sample synthetic rows, save CSV and run analysis on baseline & synthetic sets.
	- Behavior: loads a saved model, reads baseline CSV, samples synthetic data of equal length, writes synthetic CSV, and calls `analysis_utils` functions for plotting.
	- Notes: ensure the CTGAN library used here matches the model training script's API.

- `invertedPendulum.py`
	- Purpose: generate random-action data from Gym `InvertedPendulum-v5`, save CSVs and produce plots.
	- Key functions:
		- `generate_inverted_pendulum_data(num_episodes=1)` — runs episodes with random actions and returns a DataFrame.
		- `analyze_pendulum_data(num_episodes, filepath)` — reads CSV and makes plots.
		- `plot_rewards_per_episode(num_episodes, filepath)` — bar plot of total reward per episode.
	- Notes: uses `gymnasium`, `numpy`, `pandas`, `matplotlib`, and `seaborn`. Paths to `csv` and `plots` are relative; be careful with the working directory when running.

- `ppobaseline.py`
	- Purpose: train a PPO agent (`stable_baselines3`), generate trajectories with the trained agent, save CSV and run analysis.
	- Key functions:
		- `train_agent(env)` — trains PPO for `total_timesteps`.
		- `generate_data_with_trained_agent(env, model, num_episodes=50)` — runs episodes using `model.predict` and collects transitions into a DataFrame.
	- Notes: saves CSVs named with `num_episodes` and `total_timesteps`. Training with many timesteps and running many episodes can be time-consuming.

## Cross-file notes and suggestions

- Dependency mismatch: `ctgan_model.py` and `ctgan_data.py` use different CTGAN imports (`ctgan` vs `sdv.tabular`). Choose one library and update both scripts to match. Recommended: `pip install sdv` and use `from sdv.tabular import CTGAN` everywhere, or use the standalone `ctgan` package consistently.

- Output paths: several scripts save plots to `../plots` but create `plots` locally. Standardize to a single `plots` directory and pass it as a parameter to plotting functions.

- `.gitignore` currently ignores `./csv` and `./plots`. That caused git push/rebase problems because those folders exist locally. If you want to keep generated data out of version control, leave `.gitignore` as-is; otherwise remove entries and commit the generated files explicitly.

- Model save/load API: ensure the model saving/loading methods match the chosen CTGAN library (SDV vs ctgan). SDV's `CTGAN` has `.save()`/.load() methods; the `ctgan` package uses `CTGANSynthesizer` and you may need to use `pickle` for saving.

## Running the project (quick tips)

- Install dependencies in your virtualenv:

```bash
pip install gymnasium stable-baselines3 pandas matplotlib seaborn sdv
```

- Generate random data:

```bash
cd src
python3 invertedPendulum.py
```

- Train PPO agent and generate agent data:

```bash
cd src
python3 ppobaseline.py
```

- Train CTGAN and create synthetic data (after training data exists):

```bash
cd src
python3 ctgan_model.py
python3 ctgan_data.py
```

If you'd like, I can make the CTGAN API consistent across files, standardize paths and add a `requirements.txt`. Tell me which CTGAN library you prefer (SDV or ctgan) and whether you want generated `csv`/`plots` committed to the repo or kept ignored.
