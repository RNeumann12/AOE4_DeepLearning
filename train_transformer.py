"""
Training script for AoE4 transformer win prediction.

Usage example:
  python train_transformer.py --csv transformer_input.csv --epochs 10 --batch_size 64 --device cuda
"""
import argparse
import os
import random
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

import wandb

from data_transformer import build_vocabs, AoEEventDataset, collate_fn
from model import AoETransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_game_ids_from_dataset(dataset):
    # dataset.examples list has 'game_id' entries
    return [ex['game_id'] for ex in dataset.examples]


def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    losses = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = model(
                batch['entity_ids'], 
                batch['event_ids'], 
                batch['times'], 
                batch['attention_mask'], 
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ'] 
            )
            loss = loss_fn(logits, batch['labels'])
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(list(probs))
            trues.extend(list(batch['labels'].cpu().numpy()))
    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = float('nan')
    preds_bin = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(trues, preds_bin)
    return np.mean(losses), auc, acc


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # optional Weights & Biases run (guarded)
    run = None
    if args.use_wandb:
        # build W&B config; default to CLI args when not explicitly overridden
        wb_cfg = {
            "learning_rate": args.wandb_learning_rate if args.wandb_learning_rate is not None else args.lr,
            "architecture": "Transformer",
            "dataset": "AoE4",
            "epochs": args.wandb_epochs if args.wandb_epochs is not None else args.epochs,
        }
        if wandb is None or not hasattr(wandb, 'init'):
            print("Warning: wandb unavailable or missing 'init'; disabling W&B logging.")
            lr = args.lr
            epochs = args.epochs
        else:
            try:
                run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    config=wb_cfg,
                )
                wb_config = wandb.config
                lr = float(wb_config.get('learning_rate', args.lr))
                epochs = int(wb_config.get('epochs', args.epochs))
            except Exception as e:
                print(f"Warning: failed to initialize wandb: {e}; continuing without W&B.")
                run = None
                lr = args.lr
                epochs = args.epochs
    else:
        lr = args.lr
        epochs = args.epochs

    print('Building vocabs...')
    df = pd.read_csv(args.csv)
    vocabs = build_vocabs(df)

    # Determine longest sequence in dataset and decide a safe positional length / truncation
    grouped = df.groupby(['game_id', 'profile_id']).size()
    dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0
    if args.max_len is not None:
        desired_max_len = args.max_len
    else:
        desired_max_len = min(dataset_longest, args.max_pos_embed_cap)
        if dataset_longest > desired_max_len:
            print(f"Warning: longest sequence in data is {dataset_longest}, capping positional length to {desired_max_len}. Sequences will be truncated per strategy '{args.truncation_strategy}'")
            print("Tip: increase --max_pos_embed_cap, lower --batch_size, or enable --use_amp/--grad_accum_steps to avoid OOM")

    print('Creating dataset...')
    # Pass truncation strategy so we keep head+tail (or other) rather than naively truncating only head
    dataset = AoEEventDataset(args.csv, vocabs['entity_vocab'], vocabs['event_vocab'], vocabs['civ_vocab'], max_len=desired_max_len, truncation_strategy=args.truncation_strategy)

    # Use the same length for model positional embeddings
    model_max_len = desired_max_len if desired_max_len > 0 else 1

    # split by game_id
    all_game_ids = list(set(get_game_ids_from_dataset(dataset)))
    train_games, val_games = train_test_split(all_game_ids, test_size=args.val_split, random_state=args.seed)
    train_idx = [i for i, ex in enumerate(dataset.examples) if ex['game_id'] in train_games]
    val_idx = [i for i, ex in enumerate(dataset.examples) if ex['game_id'] in val_games]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}')

    model = AoETransformer(vocab_size_entity=len(vocabs['entity_vocab']), vocab_size_event=len(vocabs['event_vocab']), civ_vocab_size=len(vocabs['civ_vocab']), d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.ffn_dim, dropout=args.dropout, max_len=model_max_len)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # AMP and gradient accumulation settings -> without the GPU we use may run into OOM
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    grad_accum_steps = max(1, args.grad_accum_steps)

    # Use the effective learning rate and epoch count (which may be overridden by W&B)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=max(1, len(train_loader) // grad_accum_steps),
        epochs=epochs,
        pct_start=0.3  # Warm up for first 30% of training
    )

    patience = 5
    no_improve = 0
    best_val_auc = -1
    
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        epoch_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward / backward (with AMP optional)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(
                        batch['entity_ids'], 
                        batch['event_ids'], 
                        batch['times'], 
                        batch['attention_mask'], 
                        player_civ=batch['player_civ'],
                        enemy_civ=batch['enemy_civ']
                    )
                    loss = loss_fn(logits, batch['labels'])
            else:
                logits = model(
                    batch['entity_ids'], 
                    batch['event_ids'], 
                    batch['times'], 
                    batch['attention_mask'], 
                    player_civ=batch['player_civ'],
                    enemy_civ=batch['enemy_civ']
                )
                loss = loss_fn(logits, batch['labels'])

            raw_loss = loss.item()
            loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_losses.append(raw_loss)
            pbar.set_postfix(loss=np.mean(epoch_losses))

        val_loss, val_auc, val_acc = evaluate(model, val_loader, device)
        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            no_improve = 0
            out_path = args.output or 'best_model.pt'

            torch.save({'model_state': model.state_dict(), 'vocabs': vocabs}, out_path)
         # keep best model
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping (no improvement).")
                break
        print(f'Epoch {epoch}: train_loss={np.mean(epoch_losses):.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}')

        if run is not None:
            run.log({"epoch": epoch, "train_loss": np.mean(epoch_losses), "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc})

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            out_path = args.output or 'best_model.pt'
            torch.save({'model_state': model.state_dict(), 'vocabs': vocabs}, out_path)
            print(f'Saved best model to {out_path} (val_auc={val_auc:.4f})')
            if run is not None:
                try:
                    wandb.save(out_path)
                except Exception:
                    pass
                run.summary['best_val_auc'] = val_auc

    print('Training finished.')
    if run is not None:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='transformer_input.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='best_model.pt')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=None, help='Positional embedding length; if omitted, use dataset max sequence length')
    parser.add_argument('--max_pos_embed_cap', type=int, default=1024, help='Safety cap for positional embeddings to avoid OOM when using very long sequences')
    parser.add_argument('--truncation_strategy', choices=['head','tail','head_tail'], default='head_tail', help='How to truncate sequences when capped')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of steps to accumulate gradients before stepping optimizer')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision (AMP) to reduce memory')
    parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', default=True, help='Enable Weights & Biases logging (default: True)')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default='rineumann-universit-t-klagenfurt', help='W&B entity')
    parser.add_argument('--wandb_project', type=str, default='DeepLearning', help='W&B project')
    parser.add_argument('--wandb_epochs', type=int, default=None, help='Override epochs in W&B config')
    parser.add_argument('--wandb_learning_rate', type=float, default=None, help='Override learning rate in W&B config')
    args = parser.parse_args()
    main(args)
