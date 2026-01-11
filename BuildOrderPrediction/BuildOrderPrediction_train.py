"""
Train BuildOrderGenerator from sequences using supervised teacher forcing.

Usage:
  python train_buildorder.py --csv transformer_input.csv --epochs 10 --batch_size 64 --device cuda
"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from aoe_player_game_datset import build_vocabs, AoEEventDataset, collate_fn
from BuildOrderTransformerModel import BuildOrderGenerator, BuildOrderTrainer, load_pretrained_win_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_civ_entity_mask(dataset, entity_vocab, num_civs):
    """Construct a civ -> entity boolean mask from dataset examples.

    Args:
        dataset: AoEEventDataset instance
        entity_vocab: mapping token->id
        num_civs: number of civ ids (including pad/UNK)

    Returns:
        mask: torch.BoolTensor shape (num_civs, vocab_size_entity)
    """
    vocab_size = len(entity_vocab) + 1  # keep consistent with model + padding
    mask = torch.zeros((num_civs, vocab_size), dtype=torch.bool)

    # dataset.examples stores raw tokens and player_civ as strings; we need to map via vocab
    # The AoEEventDataset already encodes in __getitem__ but we avoid loading all items
    for ex in dataset.examples:
        p_civ = ex['player_civ']
        # map civ token (string) to index is done by build_vocabs; here dataset stores raw civ token
        # The dataset when constructed had civ_vocab mapping; we can use AoEEventDataset.civ_vocab
        civ_id = dataset.civ_vocab.get(p_civ, dataset.civ_vocab.get('<UNK>', 1))
        for ent in ex['entities']:
            ent_id = dataset.entity_vocab.get(ent, dataset.entity_vocab.get('<UNK>', 1))
            if ent_id < vocab_size:
                mask[civ_id, ent_id] = True

    print("Civ-Entity mask built:", mask)
    return mask


def evaluate(trainer, dataloader, device):
    trainer.model.eval()
    losses = []
    win_preds = []
    win_trues = []

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # derive mask if available
            if hasattr(trainer.model, 'civ_entity_mask') and trainer.model.civ_entity_mask is not None:
                allowed = trainer.model.civ_entity_mask[batch['player_civ']]
            else:
                allowed = None

            preds = trainer.model(
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ'],
                target_sequence_length=batch['entity_ids'].size(1),
                target_win_prob=batch.get('labels', None),
                teacher_forcing_ratio=1.0,  # use teacher forcing for supervised loss
                ground_truth=(batch['entity_ids'], batch['event_ids'], batch['times']),
                allowed_entities_mask=allowed
            )
            loss, _ = trainer.compute_loss(preds, (batch['entity_ids'], batch['event_ids'], batch['times']), preds[-1], batch.get('labels', None))
            losses.append(loss.item())

            # collect win predictions for metric calculation
            if 'labels' in batch and batch['labels'] is not None:
                logits = preds[-1]
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                win_preds.extend(probs)
                win_trues.extend(batch['labels'].cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else float('nan')
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(win_trues, win_preds) if len(win_trues) > 0 else float('nan')
        acc = accuracy_score(win_trues, [1 if p >= 0.5 else 0 for p in win_preds]) if len(win_trues) > 0 else float('nan')
    except Exception:
        auc = float('nan')
        acc = float('nan')

    return avg_loss, auc, acc


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    run = None
    if args.use_wandb:
        wb_cfg = {'learning_rate': args.lr, 'architecture': 'BuildOrderGenerator', 'dataset': os.path.basename(args.csv), 'epochs': args.epochs, 'batch_size': args.batch_size}
        try:
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=wb_cfg)
            wb_cfg = wandb.config
        except Exception as e:
            print(f"Warning: failed to initialize wandb: {e}. Continuing without W&B.")
            run = None

    print('Building vocabs...')
    import pandas as pd
    df = pd.read_csv(args.csv)
    vocabs = build_vocabs(df)

    print('Creating dataset...')
    grouped = df.groupby(['game_id', 'profile_id']).size()
    dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0
    desired_max_len = args.max_len if args.max_len is not None else min(dataset_longest, args.max_pos_embed_cap)

    dataset = AoEEventDataset(args.csv, vocabs['entity_vocab'], vocabs['event_vocab'], vocabs['civ_vocab'], max_len=desired_max_len, truncation_strategy=args.truncation_strategy)

    model_max_len = desired_max_len if desired_max_len > 0 else 1

    # split by game id
    all_game_ids = list({ex['game_id'] for ex in dataset.examples})
    train_games, val_games = train_test_split(all_game_ids, test_size=args.val_split, random_state=args.seed)
    train_idx = [i for i, ex in enumerate(dataset.examples) if ex['game_id'] in train_games]
    val_idx = [i for i, ex in enumerate(dataset.examples) if ex['game_id'] in val_games]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}')

    # build model
    model = BuildOrderGenerator(
        vocab_size_entity=len(vocabs['entity_vocab']) + 1,
        vocab_size_event=len(vocabs['event_vocab']) + 1,
        civ_vocab_size=len(vocabs['civ_vocab']) + 1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        max_len=model_max_len
    ).to(device)

    # create civ->entity mask and register
    civ_mask = build_civ_entity_mask(dataset, vocabs['entity_vocab'], num_civs=len(vocabs['civ_vocab']) + 1)
    model.set_civ_entity_mask(civ_mask.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = BuildOrderTrainer(model, optimizer, device=device, use_amp=args.use_amp, grad_accum_steps=args.grad_accum_steps)

    best_val_loss = float('inf')
    remediation_attempts = 0
    max_remediations = 3
    epoch = 1
    while epoch <= args.epochs:
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        epoch_losses = []
        epoch_success = True
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            try:
                loss_dict = trainer.train_step(batch, teacher_forcing_ratio=args.teacher_forcing)
            except RuntimeError as e:
                if 'out-of-memory' in str(e).lower() and remediation_attempts < max_remediations:
                    remediation_attempts += 1
                    print(f"OOM detected during training step (attempt {remediation_attempts}/{max_remediations}). Trying remediation...")
                    # Try enabling AMP if CUDA is available and not already enabled
                    hip_present = getattr(torch.version, 'hip', None) is not None
                    if torch.cuda.is_available() or hip_present:
                        if not trainer.use_amp:
                            # Enable AMP using bfloat16 on ROCm/HIP, float16 on CUDA
                            trainer.use_amp = True
                            trainer.amp_dtype = torch.bfloat16 if hip_present else torch.float16
                            trainer.amp_device = 'cuda' if (torch.cuda.is_available() or hip_present) else 'cpu'
                            try:
                                trainer.scaler = torch.amp.GradScaler(device_type=trainer.amp_device)
                            except TypeError:
                                # Fallback for older PyTorch
                                if trainer.amp_device == 'cuda':
                                    trainer.scaler = torch.cuda.amp.GradScaler()
                                else:
                                    trainer.scaler = None
                            print(f"Enabling AMP (device={trainer.amp_device}, dtype={trainer.amp_dtype}) to reduce memory usage.")
                        else:
                            print("AMP already enabled; continuing.")
                    else:
                        print("Skipping AMP enablement: CUDA/ROCm unavailable or AMP already enabled.")
                    # Halve batch size and recreate train_loader
                    old_bs = getattr(train_loader, 'batch_size', args.batch_size)
                    new_bs = max(1, int(old_bs // 2))
                    if new_bs == old_bs:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        raise RuntimeError("CUDA OOM and cannot reduce batch size further. Consider lowering --batch_size or model size.") from e
                    print(f"Reducing batch size from {old_bs} to {new_bs} and retrying epoch {epoch}.")
                    train_loader = DataLoader(train_ds, batch_size=new_bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    epoch_success = False
                    break
                else:
                    raise
            epoch_losses.append(loss_dict['total_loss'])
            pbar.set_postfix(loss=float(np.mean(epoch_losses)))

        if not epoch_success:
            # retry same epoch with modified settings
            continue

        val_loss, val_auc, val_acc = evaluate(trainer, val_loader, device)
        print(f'Epoch {epoch}: train_loss={float(np.mean(epoch_losses)):.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}')

        if run is not None:
            run.log({'epoch': epoch, 'train_loss': float(np.mean(epoch_losses)), 'val_loss': val_loss, 'val_auc': val_auc, 'val_acc': val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = args.output
            torch.save({'model_state': model.state_dict(), 'vocabs': vocabs}, out_path)
            print(f'Saved best model to {out_path} (val_loss={val_loss:.4f})')
            if run is not None:
                try:
                    wandb.save(out_path)
                    run.summary['best_val_loss'] = val_loss
                except Exception:
                    pass
        epoch += 1

    print('Training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='transformer_input.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='best_buildorder_model.pt')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--max_pos_embed_cap', type=int, default=1024)
    parser.add_argument('--truncation_strategy', choices=['head','tail','head_tail'], default='head_tail')
    parser.add_argument('--teacher_forcing', type=float, default=1.0, help='Fraction of teacher forcing during training (0-1)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of steps to accumulate gradients before stepping optimizer')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision (AMP) to reduce memory')
    parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', default=True, help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/username')
    parser.add_argument('--wandb_project', type=str, default='DeepLearning-BuildOrder', help='W&B project name')
    args = parser.parse_args()
    main(args)
