#!/usr/bin/env python3
"""
LoRA fine-tuning for the MoE WithDecoder SequencePredictor.

Freezes the base model and trains only small rank-decomposed adapter matrices
inserted into target linear layers. This lets the model adapt to a new balance
patch using 200-1500 games in 20-30 epochs instead of full retraining.

Example usage:
    python BuildOrderPrediction/MoE_WithDecoder_lora_train.py \
        --base_checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth \
        --csv_path input_data_2026_02.csv \
        --rank 8 --alpha 16.0 --target_modules ffn_only \
        --epochs 80 --lr 3e-4 --wins_only \
        --patch_id "patch_15.3.8338" \
        --output BuildOrderPrediction/lora_patch_15.3.8338.pth

Target module presets:
    ffn_only  — FFN up/down proj + output head (~1.04M params at rank=8). Default.
                Use for minor patches (stat tweaks, cost adjustments).
    attn_ffn  — ffn_only + self_attn.out_proj + cross_attn.out_proj (~1.43M params).
                Use for major patches that shift strategic meta.
    head_only — Only the output head layers (~50K params).
                Use when very few post-patch games are available (<500).
    full      — attn_ffn + map_cross_attn out_proj (~1.63M params).
                Use for complete overhauls.
    custom    — Comma-separated explicit module paths, e.g.:
                "decoder_layers.0.ffn.0,decoder_layers.0.ffn.3,entity_classifier"
"""

import os
import sys
import math
import argparse
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from BuildOrderPrediction.MoE_WithDecoder_train import (
    SequencePredictor,
    FocalLoss,
    create_data_loaders,
    compute_entity_class_weights,
)


# ──────────────────────────────────────────────────────────────────────────────
# LoRA Core
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter delta.

    Forward computes:  W·x + (alpha/rank) * B·A·x

    where:
      - W is the frozen base weight (original nn.Linear kept as sub-module)
      - A is (rank, in_features), initialized Kaiming uniform
      - B is (out_features, rank), initialized to zeros

    B is zero-initialized so the adapter output is exactly zero at step 0,
    meaning the model starts from the pre-trained state unchanged.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_linear = base_linear  # frozen; its params will be requires_grad=False
        self.rank = rank
        self.scaling = alpha / rank

        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank, device=device))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        # (x @ A.T) @ B.T  →  (..., out_features)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.base_linear.in_features}, "
            f"out={self.base_linear.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


def build_lora_target_modules(preset: str, num_decoder_layers: int = 12) -> List[str]:
    """
    Expand a preset name into a list of module dot-paths for apply_lora_to_model.

    Presets:
        ffn_only  — FFN layers + output head (default; best for minor patches)
        attn_ffn  — ffn_only + attention out_proj (for major patches)
        head_only — output head only (for very sparse data)
        full      — attn_ffn + map cross-attention out_proj
        <other>   — treated as comma-separated explicit module paths

    Args:
        preset: one of the preset names above, or comma-separated explicit paths
        num_decoder_layers: number of decoder layers in the base model

    Returns:
        List of dot-path strings, e.g. ["decoder_layers.0.ffn.0", ...]
    """
    known_presets = ("ffn_only", "attn_ffn", "head_only", "full")
    modules: List[str] = []

    if preset in ("ffn_only", "attn_ffn", "full"):
        for i in range(num_decoder_layers):
            # ffn[0] = Linear(d_model, dim_ff), ffn[3] = Linear(dim_ff, d_model)
            modules.append(f"decoder_layers.{i}.ffn.0")
            modules.append(f"decoder_layers.{i}.ffn.3")

    if preset in ("attn_ffn", "full"):
        for i in range(num_decoder_layers):
            modules.append(f"decoder_layers.{i}.self_attn.out_proj")
            modules.append(f"decoder_layers.{i}.cross_attn.out_proj")

    if preset == "full":
        for i in range(num_decoder_layers):
            modules.append(f"map_cross_attn_layers.{i}.cross_attn.out_proj")

    # Output head is included in every preset
    modules += ["entity_head.0", "entity_head.3", "entity_classifier"]

    # Custom comma-separated paths override everything above
    if preset not in known_presets:
        modules = [p.strip() for p in preset.split(",") if p.strip()]

    return modules


def apply_lora_to_model(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_modules: List[str],
) -> int:
    """
    Freeze all base model parameters, then inject LoRALinear wrappers at each
    specified module path.

    IMPORTANT: Call this AFTER loading base weights into the model. The original
    nn.Linear objects (with their loaded weights) are wrapped, not replaced.

    Args:
        model: SequencePredictor with base weights already loaded
        rank: LoRA rank r
        alpha: LoRA scaling alpha (effective scale = alpha / rank)
        target_modules: list of dot-path strings, e.g. ["decoder_layers.0.ffn.0"]

    Returns:
        Total number of newly trainable LoRA parameters
    """
    # Step 1: freeze every base parameter
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: inject LoRALinear at each target path
    lora_param_count = 0
    for path in target_modules:
        parts = path.split(".")

        # Navigate to the parent module
        parent: nn.Module = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]  # type: ignore[index]
            else:
                parent = getattr(parent, part)

        attr = parts[-1]

        # Retrieve the target layer
        if attr.isdigit():
            target = parent[int(attr)]  # type: ignore[index]
        else:
            target = getattr(parent, attr)

        if not isinstance(target, nn.Linear):
            raise TypeError(
                f"Target '{path}' is {type(target).__name__}, expected nn.Linear. "
                "Check your --target_modules preset or custom paths."
            )

        # Wrap with LoRA
        wrapper = LoRALinear(target, rank=rank, alpha=alpha)
        wrapper.lora_A.requires_grad = True
        wrapper.lora_B.requires_grad = True

        # Replace in parent — nn.Sequential uses __setitem__, modules use setattr
        if attr.isdigit():
            parent[int(attr)] = wrapper  # type: ignore[index]
        else:
            setattr(parent, attr, wrapper)

        lora_param_count += wrapper.lora_A.numel() + wrapper.lora_B.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[LoRA] Injected {len(target_modules)} adapters  |  "
        f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return lora_param_count


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only the LoRA A and B parameter tensors from the full state dict.

    The returned dict contains keys like:
        decoder_layers.0.ffn.0.lora_A
        decoder_layers.0.ffn.0.lora_B
        entity_classifier.lora_A
        ...

    Base weights (e.g. decoder_layers.0.ffn.0.base_linear.weight) are excluded.
    This is what gets saved in the LoRA checkpoint — NOT the full model weights.
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if ".lora_A" in k or ".lora_B" in k
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _filter_csv_by_patch(csv_path: str, patch_id: str) -> str:
    """
    Pre-filter a CSV to rows matching the given patch_id and write to a temp file.

    Requires a 'patch' column in the CSV. Returns the path to the filtered temp file.
    The caller is responsible for deleting the temp file if needed.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "patch" not in df.columns:
        print(
            f"[LoRA] WARNING: --patch_id '{patch_id}' specified but CSV has no 'patch' column. "
            "Using all rows."
        )
        return csv_path

    filtered = df[df["patch"].astype(str) == str(patch_id)]
    if len(filtered) == 0:
        raise ValueError(
            f"No rows found with patch=='{patch_id}' in {csv_path}. "
            f"Available values: {df['patch'].unique()[:10].tolist()}"
        )

    n_games = filtered["game_id"].nunique() if "game_id" in filtered.columns else "?"
    print(f"[LoRA] Patch filter: kept {len(filtered):,} rows from {n_games} games "
          f"(patch='{patch_id}')")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="lora_patch_"
    )
    filtered.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _run_validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: FocalLoss,
    device: torch.device,
) -> float:
    """Compute mean focal loss over the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            entity_ids = batch["entity_ids"].to(device)
            player_civ = batch["player_civ"].to(device)
            enemy_civ = batch["enemy_civ"].to(device)
            map_id = batch["map_id"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            entity_input = entity_ids[:, :-1].contiguous()
            entity_targets = entity_ids[:, 1:].contiguous()
            input_mask = mask[:, :-1].contiguous() if mask is not None else None

            if mask is not None:
                target_mask = mask[:, 1:].contiguous()
                entity_targets = entity_targets.clone()
                entity_targets[~target_mask] = -100

            entity_logits = model(
                entity_sequence=entity_input,
                player_civ=player_civ,
                enemy_civ=enemy_civ,
                map_id=map_id,
                attention_mask=input_mask,
                predict_next=False,
            )

            # FocalLoss with reduction='none' → mean manually over valid tokens
            loss_per_elem = criterion(entity_logits, entity_targets)
            valid_count = (entity_targets != -100).sum().item()
            if valid_count > 0:
                total_loss += loss_per_elem.sum().item()
                total_tokens += valid_count

    model.train()
    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for MoE WithDecoder SequencePredictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--base_checkpoint", type=str, required=True,
        help="Path to the base model checkpoint .pth file",
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to the training CSV (ideally patch-specific data)",
    )

    # LoRA config
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank r. 4=light, 8=default, 16=heavy.")
    parser.add_argument("--alpha", type=float, default=16.0,
                        help="LoRA scaling alpha. Effective scale = alpha/rank.")
    parser.add_argument(
        "--target_modules", type=str, default="ffn_only",
        help=(
            "Preset name (ffn_only|attn_ffn|full|head_only) or "
            "comma-separated explicit module paths."
        ),
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for LoRA parameters.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping.")

    # Data options (mirror base training defaults)
    parser.add_argument("--wins_only", action="store_true",
                        help="Only train on games where the player won.")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument(
        "--patch_id", type=str, default=None,
        help=(
            "If set, pre-filter the CSV to rows where the 'patch' column "
            "equals this value before training."
        ),
    )

    # Output
    parser.add_argument("--output", type=str, default="lora_adapter.pth",
                        help="Path to save the LoRA-only checkpoint.")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda.")

    return parser.parse_args()


def main() -> None:
    args = get_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[LoRA] Using device: {device}")

    torch.manual_seed(args.seed)

    # ── Load base checkpoint ──────────────────────────────────────────────────
    print(f"[LoRA] Loading base checkpoint: {args.base_checkpoint}")
    base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)

    entity_vocab: Dict[str, int] = base_ckpt["entity_vocab"]
    civ_vocab: Dict[str, int] = base_ckpt["civ_vocab"]
    map_vocab: Dict[str, int] = base_ckpt["map_vocab"]
    civ_entity_mapping = base_ckpt.get("civ_entity_mapping", None)
    base_args: dict = base_ckpt.get("args", {})

    if civ_entity_mapping is not None:
        civ_entity_mapping = {k: set(v) for k, v in civ_entity_mapping.items()}

    # ── Build model and load base weights ─────────────────────────────────────
    # dropout=0.0 during LoRA fine-tuning: dataset is small, dropout adds variance
    model = SequencePredictor(
        vocab_size_entity=len(entity_vocab),
        civ_vocab_size=len(civ_vocab),
        map_vocab_size=len(map_vocab),
        d_model=base_args.get("d_model", 1024),
        nhead=base_args.get("nhead", 16),
        num_decoder_layers=base_args.get("num_decoder_layers", 12),
        dim_feedforward=base_args.get("dim_feedforward", 4096),
        dropout=0.0,
        max_seq_len=base_args.get("max_seq_len", 50),
        num_experts=base_args.get("num_experts", 8),
        use_moe=base_args.get("use_moe", True),
        use_ngram=base_args.get("use_ngram", True),
        use_rope=base_args.get("use_rope", True),
    ).to(device)

    # BASE WEIGHTS MUST BE LOADED BEFORE LoRA INJECTION.
    # After injection, nn.Linear keys become base_linear.weight so loading fails.
    model.load_state_dict(base_ckpt["model_state_dict"])
    print(
        f"[LoRA] Base model loaded. "
        f"Total params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # ── Inject LoRA adapters (AFTER base weights are loaded) ──────────────────
    num_decoder_layers: int = base_args.get("num_decoder_layers", 12)
    target_modules = build_lora_target_modules(args.target_modules, num_decoder_layers)
    print(f"[LoRA] Target modules ({len(target_modules)} layers):")
    for m in target_modules:
        print(f"  {m}")

    apply_lora_to_model(model, rank=args.rank, alpha=args.alpha, target_modules=target_modules)

    # ── Prepare training data ─────────────────────────────────────────────────
    csv_path = args.csv_path
    tmp_csv_path: Optional[str] = None

    if args.patch_id is not None:
        tmp_csv_path = _filter_csv_by_patch(csv_path, args.patch_id)
        csv_path = tmp_csv_path

    filter_events = ["DESTROY"]
    filter_entities = ["Sheep"]

    train_loader, val_loader = create_data_loaders(
        csv_path=csv_path,
        entity_vocab=entity_vocab,
        civ_vocab=civ_vocab,
        map_vocab=map_vocab,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        val_split=args.val_split,
        seed=args.seed,
        filter_events=filter_events,
        filter_entities=filter_entities,
        wins_only=args.wins_only,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    entity_class_weights = compute_entity_class_weights(
        csv_path=csv_path,
        entity_vocab=entity_vocab,
        filter_events=filter_events,
        filter_entities=filter_entities,
    ).to(device)

    # reduction='none' so we can compute the mean over valid (non-padded) tokens
    criterion = FocalLoss(
        alpha=entity_class_weights,
        gamma=2.0,
        reduction="none",
        ignore_index=-100,
    )

    # ── Optimizer: only LoRA parameters ──────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # OneCycleLR: warms up then decays — well-suited for short fine-tuning runs
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    print(
        f"\n[LoRA] Starting fine-tuning: "
        f"{args.epochs} epochs, {len(train_loader)} batches/epoch, "
        f"lr={args.lr}, rank={args.rank}, alpha={args.alpha}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_lora_sd: Optional[Dict[str, torch.Tensor]] = None

    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            entity_ids = batch["entity_ids"].to(device)
            player_civ = batch["player_civ"].to(device)
            enemy_civ = batch["enemy_civ"].to(device)
            map_id = batch["map_id"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            # Teacher-forced sequence prediction: input[:-1] → predict[1:]
            entity_input = entity_ids[:, :-1].contiguous()
            entity_targets = entity_ids[:, 1:].contiguous()
            input_mask = mask[:, :-1].contiguous() if mask is not None else None

            if mask is not None:
                target_mask = mask[:, 1:].contiguous()
                entity_targets = entity_targets.clone()
                entity_targets[~target_mask] = -100

            entity_logits = model(
                entity_sequence=entity_input,
                player_civ=player_civ,
                enemy_civ=enemy_civ,
                map_id=map_id,
                attention_mask=input_mask,
                predict_next=False,
            )

            # FocalLoss with reduction='none': manually mean over valid tokens
            loss_per_elem = criterion(entity_logits, entity_targets)
            valid_count = (entity_targets != -100).sum()
            loss = loss_per_elem.sum() / valid_count.clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * valid_count.item()
            epoch_tokens += valid_count.item()

        avg_train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = _run_validation(model, val_loader, criterion, device)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1:>3}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lora_sd = get_lora_state_dict(model)
            print(f"  → New best val_loss={best_val_loss:.4f}  (saved)")

    # ── Save LoRA-only checkpoint ─────────────────────────────────────────────
    if best_lora_sd is None:
        # Fallback: save from current state if no improvement was recorded
        best_lora_sd = get_lora_state_dict(model)

    lora_checkpoint = {
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "lora_state_dict": best_lora_sd,
        "lora_config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "target_modules": target_modules,
        },
        "patch_id": args.patch_id,
        "val_metrics": {"val_loss": best_val_loss},
        "training_args": vars(args),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(lora_checkpoint, args.output)

    n_lora_tensors = len(best_lora_sd)
    n_lora_params = sum(v.numel() for v in best_lora_sd.values())
    print(f"\n[LoRA] Saved adapter to: {args.output}")
    print(f"  Tensors: {n_lora_tensors}  Params: {n_lora_params:,}  Best val_loss: {best_val_loss:.4f}")

    # Clean up temp CSV if one was created for patch filtering
    if tmp_csv_path is not None and os.path.exists(tmp_csv_path):
        os.unlink(tmp_csv_path)


if __name__ == "__main__":
    main()
