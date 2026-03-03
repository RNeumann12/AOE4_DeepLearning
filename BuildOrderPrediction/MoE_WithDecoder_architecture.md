# MoE With Decoder — Architecture (Readable Summary)

## Purpose
Compact, readable summary of the conditional decoder transformer used for build-order prediction (MoE + RoPE + gated cross-attention). This file documents the model components, how data flows through them, and the exact parameters used in the final training run.

## Final run (exact command)
```
python BuildOrderPrediction/MoE_WithDecoder_train.py \
  --wins_only --max_seq_len 50 --csv_path training_data_2026_01.csv \
  --d_model 1024 --nhead 16 --num_decoder_layers 12 --dim_feedforward 4096 \
  --batch_size 80 --grad_accum_steps 3 --dropout 0.1 \
  --num_experts 8 --use_moe --use_ngram --use_rope --use_contrastive \
  --teacher_forcing_ratio 1.0 --epochs 70
```

## High-level design (one-paragraph)
The model is a conditional decoder transformer: instead of a full encoder, we project conditions (player civ, enemy civ, map) through a small MLP into a dense memory vector and let each decoder layer cross-attend to that memory. The decoder is enhanced with RoPE positional embeddings, optional local attention, N-gram convolutional features, Mixture-of-Experts (MoE) blocks for specialization, gated cross-attention for condition fusion, and a deeper output head. Auxiliary losses include focal loss for imbalance, contrastive InfoNCE, and MoE load-balancing.

## Component summary (short)
- Entity embedding: token IDs → d_model embedding (with <BOS>/<PAD>/<UNK>). 
- Condition embeddings: player_civ, enemy_civ, map → each to d_model; concatenated and projected to a `d_model` condition memory vector via a small MLP.
- Positional encoding: RoPE by default (better relative modeling); fallback to learned positional embeddings if disabled.
- N-gram extractor: convolutional module (kernels 1..max_ngram) that augments token representations with local pattern features.
- Decoder layers: stack of causal decoder blocks (num_decoder_layers). Each block contains causal self-attention, gated cross-attention to condition memory, FFN with gated residuals, optional local attention, and optional MoE sub-layer applied on alternating layers.
- MoE: top-k (k=2) routing among `num_experts` experts, auxiliary load-balancing loss added to total loss.
- Gated cross-attention: a learnable gate that scales the contribution of condition memory (and separately for map) per layer.
- Output head: deeper MLP head with skip connections, GELU activations and dropout, then final linear classifier to entity vocab.

## Data & training notes (practical)
- Input sequences are truncated/padded to `max_seq_len` (here 50 seconds/steps). A leading `<BOS>` token signals sequence start.
- `filter_events=['DESTROY']` and `filter_entities=['Sheep']` were used when building vocabs and datasets for training (train script applies these by default in earlier code).
- Class imbalance is handled by focal loss, label smoothing, and log-dampened inverse-frequency class weights computed from the CSV.
- Contrastive loss (InfoNCE, temperature 0.07) is used as a small auxiliary objective (weight controlled by training args).

## Typical per-layer flow (short)
1. Token embedding + RoPE applied.
2. Add N-gram features (optional) and dropout.
3. Causal self-attention (masked) → residual/gated add.
4. Gated cross-attention to condition memory (and map-specific gated attention) → residual.
5. FFN (with gated residual) and optional MoE sub-layer → residual.

## Hyperparameters used in the final run (explicit values)
- `d_model`: 1024
- `nhead`: 16
- `num_decoder_layers`: 12
- `dim_feedforward`: 4096
- `max_seq_len`: 50
- `dropout`: 0.1
- `batch_size`: 80
- `grad_accum_steps`: 3
- `num_experts`: 8
- `use_moe`: enabled
- `use_ngram`: enabled
- `use_rope`: enabled
- `use_contrastive`: enabled
- `teacher_forcing_ratio`: 1.0
- `epochs`: 70

## Where to look in code (quick links)
- Model implementation: [BuildOrderPrediction/ MoE_WithDecoder implementation is in the training file and the SequencePredictor class inside `MoE_WithDecoder_train.py`](MoE_WithDecoder_train.py)
- Dataset & vocabs: [aoe_player_game_datset.py](aoe_player_game_datset.py)
- Losses & trainer: `SequencePredictorTrainer` within `MoE_WithDecoder_train.py` (handles focal/contrastive/MoE aux loss).

## Short checklist for reproducibility
- Use the exact command above to reproduce the final run.
- Ensure vocabs saved/loaded consistently (use `build_vocabularies()` outputs).
- Recreate `civ_entity_mask` and `entity_class_weights` from the CSV with the same filters.

If you'd like, I can (A) update the top-level README to reference this concise architecture summary, or (B) extract a minimal runnable example that instantiates the model with the final-run hyperparameters for quick tests. Which would you prefer?