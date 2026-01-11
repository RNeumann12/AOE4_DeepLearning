Transformer winrate model (AoE4)

This workspace contains a simple Transformer-based pipeline to predict a player's win probability from per-game, per-player event sequences.

Files:
- `data_transformer.py` — preprocessing utilities and `AoEEventDataset`.
- `model.py` — `AoETransformer` PyTorch model.
- `train_transformer.py` — training script (train/validation split by game id).
- `infer.py` — load a saved model and get predictions.
- `requirements.txt` — Python dependencies.

Quick start:
1. Install deps: `pip install -r requirements.txt`
2. Train: `python train_transformer.py --csv transformer_input.csv --epochs 5 --batch_size 64`
3. Infer: `python infer.py --model best_model.pt --csv transformer_input.csv --out preds.csv`

Notes:
- Sequences are tokenized by `entity` and `event` fields and padded/truncated to `--max_len`.
- Split is performed by `game_id` to avoid leakage between players in the same match.
