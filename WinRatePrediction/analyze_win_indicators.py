"""
Analyze which features/events are most associated with WIN.

This script mirrors the loss-focused analysis but focuses on signals
that increase the model's predicted probability of winning.

Usage:
  python WinRatePrediction/analyze_win_indicators.py --model best_model_len_50_no_destroy.pt --csv transformer_input_new.csv --max_len 50
"""
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aoe_player_game_datset import AoEEventDataset, collate_fn
from WinRatePrediction.WinRateTransformerModel import AoETransformer
from torch.utils.data import DataLoader


def analyze_event_statistics(df: pd.DataFrame, max_events: int = 30):
    """
    Compare event/ entity frequencies between winners and losers in early game.
    Returns DataFrames for events, entities, and event-entity pairs sorted by
    how strongly they are associated with WIN.
    """
    print("\n" + "=" * 60)
    print("1. STATISTICAL ANALYSIS: Event Frequency by Outcome (WIN indicators)")
    print("=" * 60)

    games = df.groupby(['game_id', 'profile_id'])

    win_events = []
    loss_events = []
    win_entities = []
    loss_entities = []
    win_pair = []
    loss_pair = []

    for (game_id, profile_id), group in games:
        early = group.head(max_events)
        won = early['player_won'].iloc[0] == 1

        events = early['event'].tolist()
        entities = early['entity'].tolist()
        pairs = [(e, ent) for e, ent in zip(events, entities)]

        if won:
            win_events.extend(events)
            win_entities.extend(entities)
            win_pair.extend(pairs)
        else:
            loss_events.extend(events)
            loss_entities.extend(entities)
            loss_pair.extend(pairs)

    win_event_counts = Counter(win_events)
    loss_event_counts = Counter(loss_events)
    win_entity_counts = Counter(win_entities)
    loss_entity_counts = Counter(loss_entities)
    win_pair_counts = Counter(win_pair)
    loss_pair_counts = Counter(loss_pair)

    total_wins = sum(win_event_counts.values())
    total_losses = sum(loss_event_counts.values())

    print("\n--- Event Type Analysis ---")
    event_stats = []
    all_events = set(win_event_counts.keys()) | set(loss_event_counts.keys())

    for event in all_events:
        win_freq = win_event_counts.get(event, 0) / max(total_wins, 1)
        loss_freq = loss_event_counts.get(event, 0) / max(total_losses, 1)
        # Higher = more WIN-indicative
        ratio = win_freq / max(loss_freq, 0.0001)
        event_stats.append({
            'event': event,
            'win_count': win_event_counts.get(event, 0),
            'loss_count': loss_event_counts.get(event, 0),
            'win_freq': win_freq,
            'loss_freq': loss_freq,
            'win_loss_ratio': ratio,
            'abs_diff': win_freq - loss_freq
        })

    event_df = pd.DataFrame(event_stats).sort_values('win_loss_ratio', ascending=False)
    print("\nEvents most associated with WIN (high win/loss ratio):")
    print(event_df.head(10).to_string(index=False))

    print("\n--- Entity Type Analysis ---")
    total_win_entities = sum(win_entity_counts.values())
    total_loss_entities = sum(loss_entity_counts.values())

    entity_stats = []
    all_entities = set(win_entity_counts.keys()) | set(loss_entity_counts.keys())

    for entity in all_entities:
        win_freq = win_entity_counts.get(entity, 0) / max(total_win_entities, 1)
        loss_freq = loss_entity_counts.get(entity, 0) / max(total_loss_entities, 1)
        ratio = win_freq / max(loss_freq, 0.0001)
        entity_stats.append({
            'entity': entity,
            'win_count': win_entity_counts.get(entity, 0),
            'loss_count': loss_entity_counts.get(entity, 0),
            'win_freq': win_freq,
            'loss_freq': loss_freq,
            'win_loss_ratio': ratio,
            'abs_diff': win_freq - loss_freq
        })

    entity_df = pd.DataFrame(entity_stats).sort_values('win_loss_ratio', ascending=False)
    print("\nEntities most associated with WIN (high win/loss ratio):")
    print(entity_df.head(15).to_string(index=False))

    print("\nEntities most associated with LOSS (low win/loss ratio):")
    print(entity_df.tail(15).to_string(index=False))

    print("\n--- Event + Entity Pair Analysis ---")
    total_win_pairs = sum(win_pair_counts.values())
    total_loss_pairs = sum(loss_pair_counts.values())

    pair_stats = []
    all_pairs = set(win_pair_counts.keys()) | set(loss_pair_counts.keys())

    for pair in all_pairs:
        win_freq = win_pair_counts.get(pair, 0) / max(total_win_pairs, 1)
        loss_freq = loss_pair_counts.get(pair, 0) / max(total_loss_pairs, 1)
        ratio = win_freq / max(loss_freq, 0.0001)
        if win_pair_counts.get(pair, 0) + loss_pair_counts.get(pair, 0) >= 50:
            pair_stats.append({
                'event': pair[0],
                'entity': pair[1],
                'win_count': win_pair_counts.get(pair, 0),
                'loss_count': loss_pair_counts.get(pair, 0),
                'win_freq': win_freq,
                'loss_freq': loss_freq,
                'win_loss_ratio': ratio,
                'abs_diff': win_freq - loss_freq
            })

    pair_df = pd.DataFrame(pair_stats).sort_values('win_loss_ratio', ascending=False)
    print("\nEvent+Entity pairs most associated with WIN:")
    print(pair_df.head(20).to_string(index=False))

    return event_df, entity_df, pair_df


def analyze_timing(df: pd.DataFrame, max_events: int = 30):
    print("\n" + "=" * 60)
    print("2. TIMING ANALYSIS: Event Timing by Outcome (WIN indicators)")
    print("=" * 60)

    games = df.groupby(['game_id', 'profile_id'])
    timing_stats = defaultdict(lambda: {'win_times': [], 'loss_times': []})

    for (game_id, profile_id), group in games:
        early_game = group.head(max_events)
        won = early_game['player_won'].iloc[0] == 1
        for _, row in early_game.iterrows():
            event_entity = f"{row['event']}_{row['entity']}"
            if won:
                timing_stats[event_entity]['win_times'].append(row['time'])
            else:
                timing_stats[event_entity]['loss_times'].append(row['time'])

    timing_results = []
    for event_entity, data in timing_stats.items():
        if len(data['win_times']) >= 30 and len(data['loss_times']) >= 30:
            win_mean = np.mean(data['win_times'])
            loss_mean = np.mean(data['loss_times'])
            timing_results.append({
                'event_entity': event_entity,
                'win_mean_time': win_mean,
                'loss_mean_time': loss_mean,
                'time_diff': win_mean - loss_mean,  # Positive = winners do it later
                'win_count': len(data['win_times']),
                'loss_count': len(data['loss_times'])
            })

    timing_df = pd.DataFrame(timing_results).sort_values('time_diff', ascending=False)
    print("\nEvents that winners do LATER than losers:")
    print(timing_df.head(15).to_string(index=False))
    print("\nEvents that winners do EARLIER than losers:")
    print(timing_df.tail(15).to_string(index=False))
    return timing_df


def gradient_attribution(model, dataloader, vocabs, device, num_samples=500):
    """
    Compute gradients that increase win probability (focus on positive influence).
    """
    print("\n" + "=" * 60)
    print("3. GRADIENT ATTRIBUTION: Which Events Drive WIN Predictions")
    print("=" * 60)

    model.eval()
    entity_idx_to_name = {v: k for k, v in vocabs['entity_vocab'].items()}
    event_idx_to_name = {v: k for k, v in vocabs['event_vocab'].items()}

    entity_importance = defaultdict(list)
    event_importance = defaultdict(list)
    pair_importance = defaultdict(list)

    processed = 0

    for batch in dataloader:
        if processed >= num_samples:
            break

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        entity_ids = batch['entity_ids']
        event_ids = batch['event_ids']
        times = batch['times']
        mask = batch['attention_mask']

        B, L = entity_ids.size()

        entity_embeds = model.entity_embed(entity_ids)
        event_embeds = model.event_embed(event_ids)

        entity_embeds.retain_grad()
        event_embeds.retain_grad()

        x = entity_embeds + event_embeds
        seq_positions = torch.arange(0, L, device=entity_ids.device).unsqueeze(0).expand(B, L)
        x = x + model.seq_pos_embed(seq_positions)
        x = x + model.time_encoding(times)

        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_mask = torch.ones((B, 1), device=mask.device)
        full_mask = torch.cat((cls_mask, mask), dim=1)
        src_key_padding_mask = (full_mask == 0)

        encoded = model.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = encoded[:, 0, :]

        maps = model.map_embed(batch['map'])
        p_civ = model.civ_embed(batch['player_civ'])
        e_civ = model.civ_embed(batch['enemy_civ'])

        combined = torch.cat([cls_output, p_civ, e_civ], dim=1)
        logits = model.classifier(combined).squeeze(-1)

        probs = torch.sigmoid(logits)

        # Focus on WIN predictions: maximize probs
        win_score = probs.sum()
        win_score.backward()

        if entity_embeds.grad is not None and event_embeds.grad is not None:
            entity_grad = entity_embeds.grad.abs().mean(dim=-1)
            event_grad = event_embeds.grad.abs().mean(dim=-1)

            for b in range(B):
                for pos in range(L):
                    if mask[b, pos] == 1:
                        ent_idx = entity_ids[b, pos].item()
                        evt_idx = event_ids[b, pos].item()
                        ent_name = entity_idx_to_name.get(ent_idx, '<UNK>')
                        evt_name = event_idx_to_name.get(evt_idx, '<UNK>')
                        entity_importance[ent_name].append(entity_grad[b, pos].item())
                        event_importance[evt_name].append(event_grad[b, pos].item())
                        pair_importance[f"{evt_name}_{ent_name}"].append(
                            (entity_grad[b, pos].item() + event_grad[b, pos].item()) / 2
                        )

        model.zero_grad()
        processed += B

    def aggregate(d):
        return {k: (np.mean(v), np.std(v), len(v)) for k, v in d.items() if len(v) >= 10}

    entity_agg = aggregate(entity_importance)
    event_agg = aggregate(event_importance)
    pair_agg = aggregate(pair_importance)

    print(f"\nProcessed {processed} samples")

    print("\n--- Event Types by Gradient Importance (driving WIN predictions) ---")
    event_sorted = sorted(event_agg.items(), key=lambda x: x[1][0], reverse=True)
    for name, (mean, std, count) in event_sorted[:10]:
        print(f"  {name:20s}: mean={mean:.6f}, std={std:.6f}, n={count}")

    print("\n--- Entity Types by Gradient Importance ---")
    entity_sorted = sorted(entity_agg.items(), key=lambda x: x[1][0], reverse=True)
    for name, (mean, std, count) in entity_sorted[:20]:
        print(f"  {name:30s}: mean={mean:.6f}, std={std:.6f}, n={count}")

    print("\n--- Event+Entity Pairs by Gradient Importance ---")
    pair_sorted = sorted(pair_agg.items(), key=lambda x: x[1][0], reverse=True)
    for name, (mean, std, count) in pair_sorted[:25]:
        print(f"  {name:40s}: mean={mean:.6f}, std={std:.6f}, n={count}")

    return entity_agg, event_agg, pair_agg


def permutation_importance(model, dataloader, vocabs, device, num_batches=10):
    print("\n" + "=" * 60)
    print("4. PERMUTATION IMPORTANCE: Impact of Shuffling Events (WIN)")
    print("=" * 60)

    model.eval()

    all_batches = []
    baseline_preds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            all_batches.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
            logits = model(
                batch['entity_ids'],
                batch['event_ids'],
                batch['times'],
                batch['attention_mask'],
                batch['map'],
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ']
            )
            probs = torch.sigmoid(logits)
            baseline_preds.append(probs.cpu())

    baseline_preds = torch.cat(baseline_preds)

    entity_importance_results = {}
    # Limit to a subset for speed
    all_entities = list(set(vocabs['entity_vocab'].keys()) - {'<PAD>', '<UNK>'})[:30]

    print("\nTesting entity permutation importance (top 30 entities)...")
    for entity in all_entities:
        entity_idx = vocabs['entity_vocab'].get(entity, 1)
        shuffled_preds = []
        with torch.no_grad():
            for batch in all_batches:
                entity_ids = batch['entity_ids'].clone()
                mask = (entity_ids == entity_idx)
                if mask.any():
                    random_entities = torch.randint(2, len(vocabs['entity_vocab']), entity_ids.shape, device=entity_ids.device)
                    entity_ids[mask] = random_entities[mask]
                logits = model(
                    entity_ids,
                    batch['event_ids'],
                    batch['times'],
                    batch['attention_mask'],
                    batch['map'],
                    player_civ=batch['player_civ'],
                    enemy_civ=batch['enemy_civ']
                )
                probs = torch.sigmoid(logits)
                shuffled_preds.append(probs.cpu())
        shuffled_preds = torch.cat(shuffled_preds)
        importance = (baseline_preds - shuffled_preds).abs().mean().item()
        entity_importance_results[entity] = importance

    sorted_entities = sorted(entity_importance_results.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Entities by Permutation Importance (prediction change when shuffled) ---")
    for name, importance in sorted_entities[:20]:
        print(f"  {name:30s}: {importance:.6f}")

    return entity_importance_results


def analyze_position_importance(model, dataloader, device, num_samples=500):
    print("\n" + "=" * 60)
    print("5. POSITION IMPORTANCE: Which Sequence Positions Matter Most (WIN)")
    print("=" * 60)

    model.eval()
    position_importance = defaultdict(list)
    processed = 0

    for batch in dataloader:
        if processed >= num_samples:
            break
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        entity_ids = batch['entity_ids']
        event_ids = batch['event_ids']
        B, L = entity_ids.size()

        with torch.no_grad():
            baseline_logits = model(
                entity_ids,
                event_ids,
                batch['times'],
                batch['attention_mask'],
                batch['map'],
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ']
            )
            baseline_probs = torch.sigmoid(baseline_logits)

        for pos in range(min(L, 30)):
            e_masked = entity_ids.clone()
            ev_masked = event_ids.clone()
            e_masked[:, pos] = 0
            ev_masked[:, pos] = 0
            with torch.no_grad():
                masked_logits = model(
                    e_masked,
                    ev_masked,
                    batch['times'],
                    batch['attention_mask'],
                    batch['map'],
                    player_civ=batch['player_civ'],
                    enemy_civ=batch['enemy_civ']
                )
                masked_probs = torch.sigmoid(masked_logits)
            importance = (baseline_probs - masked_probs).abs().mean().item()
            position_importance[pos].append(importance)

        processed += B

    print("\n--- Position Importance (mean prediction change when masked) ---")
    for pos in range(min(30, len(position_importance))):
        if position_importance[pos]:
            mean_imp = np.mean(position_importance[pos])
            print(f"  Position {pos:2d}: {mean_imp:.6f}")

    return position_importance


def main(args):
    print("=" * 60)
    print("WIN INDICATOR ANALYSIS")
    print("=" * 60)

    print('\nLoading model...')
    chk = torch.load(args.model, map_location='cpu')
    vocabs = chk['vocabs']

    print('Loading dataset...')
    df = pd.read_csv(args.csv)
    if 'event' in df.columns:
        df = df[df['event'] != 'DESTROY']
    print(f"Total rows (after DESTROY filter): {len(df)}, Games: {df['game_id'].nunique()}")

    grouped = df.groupby(['game_id', 'profile_id']).size()
    dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0
    desired_max_len = args.max_len if args.max_len else min(dataset_longest, 1024)

    if args.run_stats:
        event_df, entity_df, pair_df = analyze_event_statistics(df, max_events=desired_max_len)
        timing_df = analyze_timing(df, max_events=desired_max_len)

    if args.run_gradients or args.run_permutation or args.run_positions:
        ds = AoEEventDataset(
            args.csv,
            vocabs['entity_vocab'],
            vocabs['event_vocab'],
            vocabs['civ_vocab'],
            vocabs['map_vocab'],
            max_len=desired_max_len,
            truncation_strategy='head',
            filter_events=['DESTROY']
        )
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

        model_max_len = chk['model_state']['seq_pos_embed.weight'].shape[0]
        model = AoETransformer(
            vocab_size_entity=len(vocabs['entity_vocab']),
            vocab_size_event=len(vocabs['event_vocab']),
            civ_vocab_size=len(vocabs['civ_vocab']),
            map_vocab_size=len(vocabs['map_vocab']),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
            max_len=model_max_len
        )
        model.load_state_dict(chk['model_state'])

        device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        model.to(device)
        model.eval()

        if args.run_gradients:
            gradient_attribution(model, loader, vocabs, device, num_samples=args.num_samples)

        if args.run_permutation:
            permutation_importance(model, loader, vocabs, device, num_batches=20)

        if args.run_positions:
            analyze_position_importance(model, loader, device, num_samples=args.num_samples)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pt')
    parser.add_argument('--csv', type=str, default='transformer_input.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=500)

    parser.add_argument('--run_stats', action='store_true', default=True)
    parser.add_argument('--run_gradients', action='store_true', default=True)
    parser.add_argument('--run_permutation', action='store_true', default=False)
    parser.add_argument('--run_positions', action='store_true', default=False)
    parser.add_argument('--no_stats', action='store_false', dest='run_stats')
    parser.add_argument('--no_gradients', action='store_false', dest='run_gradients')

    args = parser.parse_args()
    main(args)
