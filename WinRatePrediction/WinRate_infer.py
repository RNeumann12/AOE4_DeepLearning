"""
Inference script: load saved model and run on dataset, printing / saving predictions.

Usage examples:
  python infer.py --model best_model.pt --csv transformer_input.csv --out preds.csv
"""
import argparse
import torch
import pandas as pd
from aoe_player_game_datset import build_vocabs, AoEEventDataset, collate_fn
from WinRatePrediction.WinRateTransformerModel import AoETransformer
from torch.utils.data import DataLoader


def main(args):
    print('Loading saved model...')
    data_df = pd.read_csv(args.csv)
    vocabs = build_vocabs(data_df)

    # Determine longest sequence and apply optional safety cap
    grouped = pd.read_csv(args.csv).groupby(['game_id','profile_id']).size()
    dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0
    if args.max_len is not None:
        desired_max_len = args.max_len
    else:
        desired_max_len = min(dataset_longest, args.max_pos_embed_cap)
        if dataset_longest > desired_max_len:
            print(f"Warning: longest sequence in data is {dataset_longest}, capping positional length to {desired_max_len}. Sequences will be truncated per strategy '{args.truncation_strategy}'")

    ds = AoEEventDataset(args.csv, vocabs['entity_vocab'], vocabs['event_vocab'], vocabs['civ_vocab'], max_len=desired_max_len, truncation_strategy=args.truncation_strategy)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # If max_len not specified, set model positional embedding size to longest sequence in dataset
    model_max_len = desired_max_len if desired_max_len > 0 else 1

    chk = torch.load(args.model, map_location='cpu')
    model = AoETransformer(vocab_size_entity=len(vocabs['entity_vocab']), vocab_size_event=len(vocabs['event_vocab']), civ_vocab_size=len(vocabs['civ_vocab']), d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.ffn_dim, dropout=args.dropout, max_len=model_max_len)
    model.load_state_dict(chk['model_state'])

    # Move model to selected device and set eval mode
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model.to(device)
    model.eval()

    preds = []
    indices = []
    with torch.no_grad():
        for batch in loader:
            # move tensors to device
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
            probs = torch.sigmoid(logits)
            preds.extend(list(probs.cpu().numpy()))
    # print basic stats
    import numpy as np
    print(f'Predictions: mean={np.mean(preds):.4f}, min={np.min(preds):.4f}, max={np.max(preds):.4f}')
    # Optionally save
    if args.out:
        pd.DataFrame({'pred': preds}).to_csv(args.out, index=False)
        print(f'Saved predictions to {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pt')
    parser.add_argument('--csv', type=str, default='transformer_input.csv')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=None, help='Positional embedding length; if omitted, use dataset max sequence length')
    parser.add_argument('--max_pos_embed_cap', type=int, default=1024, help='Safety cap for positional embeddings to avoid OOM when using very long sequences')
    parser.add_argument('--truncation_strategy', choices=['head','tail','head_tail'], default='head_tail', help='How to truncate sequences when capped')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on ("cuda" or "cpu"). Will fall back to CPU if CUDA unavailable')
    args = parser.parse_args()
    main(args)
