"""
Training script for AoE4 transformer win prediction.

Usage example:
    python -m WinRatePrediction.WinRate_train --csv transformer_input.csv --epochs 10 --batch_size 64 --device cuda
    python -m WinRatePrediction.WinRate_train --csv transformer_input.csv --epochs 20 --batch_size 64 --device cuda --max_len 50 --output best_model_len_50.pt

    python -m WinRatePrediction.WinRate_train --filter_destroy_events --csv transformer_input.csv --epochs 20 --batch_size 64 --device cuda --max_len 50 --output best_model_len_50_no_destroy.pt
    python -m WinRatePrediction.WinRate_train --filter_destroy_events --csv transformer_input_new.csv --epochs 20 --batch_size 64 --device cuda --max_len 200 --output best_model_len_200_no_destroy.pt

"""
import argparse
import os
import random
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, brier_score_loss
from tqdm import tqdm

# Ensure project root is on PYTHONPATH for local imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aoe_player_game_datset import AoEEventDataset

import wandb

from aoe_player_game_datset import build_vocabs, AoEEventDataset, collate_fn
from WinRatePrediction.WinRateTransformerModel import AoETransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_game_ids_from_dataset(dataset):
    # dataset.examples list has 'game_id' entries
    return [ex['game_id'] for ex in dataset.examples]


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += np.abs(bin_conf - bin_acc) * (mask.sum() / len(probs))

    return ece

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
                batch['map'], 
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ'] 
            )
            loss = loss_fn(logits, batch['labels'])
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(list(probs))
            trues.extend(list(batch['labels'].cpu().numpy()))
    
    preds = np.array(preds)
    trues = np.array(trues)

    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = float('nan')

    preds_bin = (preds >= 0.5).astype(int)
    acc = accuracy_score(trues, preds_bin)

    CONF_HIGH = 0.7   # sehr sicher win
    CONF_LOW  = 0.3   # sehr sicher loss

    tn, fp, fn, tp = confusion_matrix(trues, preds_bin).ravel()

    fpr = fp / (fp + tn + 1e-8) #false positive rate
    fnr = fn / (fn + tp + 1e-8) # false negative rate
    wprecision = tp / (tp + fp + 1e-8) # wins, when win is predicted
    lprecision = tn / (tn + fn + 1e-8) # wins, when win is predicted
    recall = tp / (tp + fn + 1e-8) # wins, actually caught
    balanced_acc = 0.5 * (
        tp / (tp + fn + 1e-8) +
        tn / (tn + fp + 1e-8)
    ) # recall for wins and losses

    confident_wrong_win = preds[(preds >= CONF_HIGH) & (trues == 0)]
    confident_wrong_loss = preds[(preds <= CONF_LOW) & (trues == 1)]
    wrong = preds[preds_bin != trues]

    f1score = 2*((wprecision*recall) / (wprecision+recall))

    ece = expected_calibration_error(preds, trues)
    brier = brier_score_loss(trues, preds)

    wandb.log({
    # --- Core metrics ---
    "val/loss": np.mean(losses),
    "val/auc": auc,
    "val/accuracy": acc,
    "val/balanced_accuracy": balanced_acc,
    "val/f1": f1score,

    # --- Confusion matrix (counts) ---
    "val/confusion/tp_win": tp,
    "val/confusion/tn_loss": tn,
    "val/confusion/fp_pred_win": fp,
    "val/confusion/fn_missed_win": fn,

    # --- Error rates ---
    "val/error_rate/fpr_pred_win": fpr,   # false win predictions
    "val/error_rate/fnr_missed_win": fnr, # missed wins

    # --- Precision / Recall ---
    "val/precision/win": wprecision,
    "val/precision/loss": lprecision,
    "val/recall/win": recall,

    # --- Confidence analysis ---
    "val/confidence/pred_distribution": wandb.Histogram(preds),

    "val/confidence/confident_wrong_win_rate":
        len(confident_wrong_win) / len(preds),

    "val/confidence/confident_wrong_loss_rate":
        len(confident_wrong_loss) / len(preds),

    "val/confidence/confident_wrong_win_hist":
        wandb.Histogram(confident_wrong_win),

    "val/confidence/confident_wrong_loss_hist":
        wandb.Histogram(confident_wrong_loss),

    "val/confidence/mean_confidence_wrong":
        np.mean(np.abs(wrong - 0.5)) * 2,

    "val/brier": brier,
    "val/expected_calibration_error": ece,
    })

    return np.mean(losses), auc, acc, brier


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
            "dataset": "AoE4-with-map",
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
    
    # Optionally filter out DESTROY events
    temp_csv_path = None
    if args.filter_destroy_events:
        import tempfile
        original_len = len(df)
        df = df[df['event'] != 'DESTROY']
        print(f"Filtered out DESTROY events: {original_len} -> {len(df)} rows ({original_len - len(df)} removed)")
        # Save filtered data to temp file so dataset loads the filtered version
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name
        csv_path = temp_csv_path
    else:
        csv_path = args.csv
    
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
    dataset = AoEEventDataset(csv_path, vocabs['entity_vocab'], vocabs['event_vocab'], vocabs['civ_vocab'], vocabs['map_vocab'],max_len=desired_max_len, truncation_strategy=args.truncation_strategy)

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

    model = AoETransformer(vocab_size_entity=len(vocabs['entity_vocab']), vocab_size_event=len(vocabs['event_vocab']), civ_vocab_size=len(vocabs['civ_vocab']), map_vocab_size=len(vocabs['map_vocab']), d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.ffn_dim, dropout=args.dropout, max_len=model_max_len)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    num_pos = 0
    num_neg = 0
    for batch in train_loader:
        labels = batch['labels']
        num_pos += (labels == 1).sum().item()
        num_neg += (labels == 0).sum().item()

    pos_weight = num_neg / max(num_pos, 1)  # avoid division by zero
    print(f"pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")

    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

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

    patience = 8
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
                        batch['map'], 
                        player_civ=batch['player_civ'],
                        enemy_civ=batch['enemy_civ']
                    )

            else:
                logits = model(
                    batch['entity_ids'], 
                    batch['event_ids'], 
                    batch['times'], 
                    batch['attention_mask'], 
                    batch['map'], 
                    player_civ=batch['player_civ'],
                    enemy_civ=batch['enemy_civ']
                )
            
            # model was overly confident, so we smooth the binary labels (basically penalize for being too confident)
            labels = batch["labels"] * 0.9 + 0.05  # simple binary label smoothing
            
            logits = torch.clamp(logits, -8, 8) # cap logits to avoid numerical dominance from a few samples.
            loss = loss_fn(logits, labels)

            # loss = loss_fn(logits, batch['labels'])

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

        val_loss, val_auc, val_acc, val_brier = evaluate(model, val_loader, device)
        print(f"ΔAUC = {val_auc - best_val_auc:.5f}")
        if val_auc > best_val_auc + 1e-3: # larger to reduce more noise
            best_val_auc = val_auc
            no_improve = 0
            out_path = args.output or 'best_model.pt'

            torch.save({'model_state': model.state_dict(), 'vocabs': vocabs}, out_path)
         # keep best model
        else:
            no_improve += 1
            print(f"patience = {patience} | no_improve = {no_improve}")
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
    
    # Clean up temp file if created
    if temp_csv_path is not None:
        os.unlink(temp_csv_path)
    
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
    parser.add_argument('--truncation_strategy', choices=['head','tail','head_tail'], default='head', help='How to truncate sequences when capped')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of steps to accumulate gradients before stepping optimizer')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision (AMP) to reduce memory')
    parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', default=True, help='Enable Weights & Biases logging (default: True)')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default='rineumann-universit-t-klagenfurt', help='W&B entity')
    parser.add_argument('--wandb_project', type=str, default='DeepLearning-WinRate', help='W&B project')
    parser.add_argument('--wandb_epochs', type=int, default=None, help='Override epochs in W&B config')
    parser.add_argument('--wandb_learning_rate', type=float, default=None, help='Override learning rate in W&B config')
    parser.add_argument('--filter_destroy_events', action='store_true', default=False, help='Filter out all DESTROY events from training data')
    args = parser.parse_args()
    main(args)
