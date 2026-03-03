# AOE4 Deep Learning

This repository implements a collection of deep-learning tools and data pipelines for analyzing Age of Empires 4 match logs. The project focuses on four interconnected components:

- Win Rate Prediction: predict match outcomes from early-game signals.
- Build Order Prediction: predict next buildings/units in a player's build order.
- Data Mining & Processing: convert raw game logs into model-ready datasets.
- Strategy Discovery: unsupervised and supervised methods to surface common strategies.

This README summarizes the project structure, core files, model design choices, and quickstart instructions for training and inference.

**Project Goals**

- Provide reliable models that can predict short-term player behavior (build order) and match outcomes (win rate).
- Offer robust data pipelines to transform raw `.jsonl` game logs into tokenized, time-aware sequences.
- Enable strategy discovery through clustering and embedding analysis of player sequences.

**Repository layout (high level)**

- `aoe_player_game_datset.py` — shared dataset class, vocabulary builders, and dataset utilities.
- `DataPreperation/` — scripts to convert raw `.jsonl` logs into CSVs used for training.
- `BuildOrderPrediction/` — models, training and inference code for build-order prediction (including MoE variants).
- `WinRatePrediction/` — models and analysis tools for win-rate prediction.
- `StrategyDiscovery/` and `StrategyDiscoverySupervised/` — clustering and supervised strategy analysis tools.
- `transformer_input.csv*`, `training_data_2026_01.csv` — example dataset artifacts and inputs.

**High-level components**

**1) Win Rate Prediction**

- Purpose: Predict the probability a player (or a match) will win from early game events and contextual features (player civ, enemy civ, map, time series of events).
- Model: Transformer encoder with time-based positional encoding (sinusoidal/time-phase). See `WinRatePrediction/WinRateTransformerModel.py`.
- Input: encoded event/entity token sequences, time features, civ/map embeddings.
- Output: binary win/loss probability.
- Notes: Designed to run on truncated early-game windows (e.g., `--max_len 50`) to evaluate early indicators.

Example: Generate winrate prediction for dataset
```bash
       python WinRatePrediction/WinRate_infer.py --model WinRatePrediction/winrate_final_model.pt --csv  TrainingData/training_data_2026-01-21.csv --max_len 200 --filter_destroy_events  --print_game_table
```

**2) Build Order Prediction**

- Purpose: Given a player's past build events and match context, predict the next entity (unit/building) they will queue/build.
- Model variants:
	- Simple encoder-decoder Transformer.
	- MoE (Mixture-of-Experts) routing per civilization for specialization.
	- MoE + Decoder for full pipeline variants.
- Loss & training tricks: focal loss for class imbalance, label smoothing, and class weights (log-dampened inverse frequency).
- Important files: `BuildOrderPrediction/BuildOrderTransformerModel.py`, `BuildOrderPrediction/BuildOrderPrediction_train.py`, `BuildOrderPrediction/MoE_*.py`.

Quick training example:
```bash
python BuildOrderPrediction/BuildOrderPrediction_train.py \
  --csv TrainingData/training_data_2026-01-21.csv \
  --epochs 50 \
  --label_smoothing 0.2 \
  --truncation_strategy head \
  --max_len 256
```

**3) Data Mining & Processing**

- Purpose: Convert raw AoE4 event logs (`.jsonl`) into per-player/per-game sequences and build shared vocabularies for entities, events, and civs.
- Key behavior:
	- Groups events by `(game_id, profile_id)`.
	- Builds three shared vocabularies (entity, event, civ) with frequency filtering.
	- Encodes sequences into token IDs and numeric/temporal features.
	- Applies domain-specific filters when building vocabs (e.g., remove `DESTROY` events, skip `Sheep`).
- Script: `DataPreperation/data_prep.py` — run this to generate CSVs like `transformer_input.csv`.

Example:
```bash
python DataPreperation/data_prep.py collected_games_with_summary_v2_2026-01-21.jsonl
```

Notes:
- Vocabulary consistency is critical — save and reuse vocabs at inference to avoid OOV mismatches.
- Time values are expected to be integer seconds and are truncated/clamped to configured `max_time_seconds` (default 5400).

**4) Strategy Discovery**

- Purpose: Extract recurring high-level strategies from player sequences using clustering, embeddings, and supervised labels.
- Tools: clustering pipelines and embedding analysis under `StrategyDiscovery/` and `StrategyDiscoverySupervised/`.
- Outputs: cluster profiles, strategy labels, and analysis visualizations (many logged to `wandb/` when available).

**Conventions & gotchas**

- Checkpoints: model files are commonly named `final_*.pt` or `*.pth` (inconsistent extensions exist in the repo).
- Filtering during vocabulary build (e.g., `DESTROY`) is applied only at training-data creation;
- Class imbalance: use focal loss (`--use_focal_loss`) and label smoothing to avoid degenerate predictions (e.g., always predicting Villager).
- Train/validation split: done at the per-game level to preserve game integrity (no random sampling of events across games).

**Quickstart (setup)**

1. Create a Python environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data (example):
```bash
python DataPreperation/data_prep.py collected_games_with_summary_v2_2026-01-21.jsonl
```

3. Train a build-order model (example):
```bash
python BuildOrderPrediction/BuildOrderPrediction_train.py --csv transformer_input_new.csv --epochs 30
```

4. Run inference / analysis examples in their respective folders (`BuildOrderPrediction/`, `WinRatePrediction/`).

**Useful files**

- `aoe_player_game_datset.py` — dataset + vocab utilities (central).
- `BuildOrderPrediction/BuildOrderTransformerModel.py` — model & loss implementations.
- `WinRatePrediction/WinRateTransformerModel.py` — win-rate model definitions.
- `DataPreperation/data_prep.py` — raw-to-csv preprocessing.

**Next steps & contributions**

- Add standardized saving/loading for vocab artifacts used in inference.
- Add example notebooks demonstrating embedding visualization and cluster interpretation.
- Better unify checkpoint naming and add a small `scripts/` helper for common runs.

---

Project maintained by the repository contributors. For questions or help, open an issue or request a walkthrough of a specific component.
