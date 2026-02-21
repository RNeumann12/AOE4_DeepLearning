#!/usr/bin/env python3
"""Infer a build order for a Civilization matchup using a trained BuildOrderGenerator.

Example (generate build order):
  python BuildOrderPrediction/BuildOrderPrediction_infer.py --ckpt best_buildorder_model.pt --player_civ English --enemy_civ French --beam_width 4 --max_length 12
  python BuildOrderPrediction/BuildOrderPrediction_infer.py --ckpt best_buildorder_model.pt --top_probs 5 --player_civ English --enemy_civ French --max_length 12 --greedy 1
  
Example (evaluate on test data to reproduce training metrics):
  python BuildOrderPrediction/BuildOrderPrediction_infer.py --ckpt best_buildorder_model.pt --eval_csv input_with_map.csv --eval_split 0.1
"""
import argparse
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import re

# Prefer package import when run as a module; fall back to adding repo root to sys.path
try:
    from BuildOrderPrediction.BuildOrderTransformerModel import BuildOrderGenerator
except ModuleNotFoundError:
    # Running the script directly (python BuildOrderPrediction/infer_buildorder.py) may
    # set the import path to the package dir instead of the repo root. Add parent dir
    # to sys.path and retry the import so the script works both as a module and a script.
    import os, sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from BuildOrderPrediction.BuildOrderTransformerModel import BuildOrderGenerator


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Invert vocabulary from token->id to id->token.
    
    Ensures special tokens are properly mapped.
    """
    inv = {v: k for k, v in vocab.items()}
    # Only set PAD if not already in vocab (don't override existing mapping)
    if 0 not in inv:
        inv[0] = '<PAD>'
    return inv


def find_civ_id(civ_name: str, civ_vocab: Dict[str, int]):
    # Try exact match, then case-insensitive
    if civ_name in civ_vocab:
        return civ_name, civ_vocab[civ_name]
    lower_map = {k.lower(): v for k, v in civ_vocab.items()}
    if civ_name.lower() in lower_map:
        return list(civ_vocab.keys())[list(lower_map.values()).index(lower_map[civ_name.lower()])], lower_map[civ_name.lower()]
    # Not found
    return None, None


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model from checkpoint and return model, vocabs, and inverted vocabs."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'vocabs' not in ckpt or 'model_state' not in ckpt:
        raise RuntimeError('Checkpoint must contain keys: "model_state" and "vocabs"')
    
    vocabs = ckpt['vocabs']
    ent_vocab = vocabs['entity_vocab']
    ev_vocab = vocabs['event_vocab']
    civ_vocab = vocabs['civ_vocab']

    inv_ent = invert_vocab(ent_vocab)
    inv_ev = invert_vocab(ev_vocab)

    # Attempt to infer architecture/d_model from saved state dict shapes
    sd = ckpt['model_state']
    if 'entity_head.weight' not in sd or 'event_head.weight' not in sd or 'civ_embed.weight' not in sd:
        raise RuntimeError('Saved state dict does not contain expected parameter names (entity_head.weight etc.)')

    vocab_size_entity = sd['entity_head.weight'].shape[0]
    d_model = sd['entity_head.weight'].shape[1]
    vocab_size_event = sd['event_head.weight'].shape[0]
    civ_vocab_size = sd['civ_embed.weight'].shape[0]

    # Infer encoder/decoder layer counts and positional embedding length from state dict
    enc_idxs = [int(m.group(1)) for k in sd.keys() for m in [re.match(r'encoder\.layers\.(\d+)\.', k)] if m]
    dec_idxs = [int(m.group(1)) for k in sd.keys() for m in [re.match(r'decoder\.layers\.(\d+)\.', k)] if m]
    num_encoder_layers = max(enc_idxs) + 1 if enc_idxs else 1
    num_decoder_layers = max(dec_idxs) + 1 if dec_idxs else 1

    seq_pos_rows = sd['seq_pos_embed.weight'].shape[0] if 'seq_pos_embed.weight' in sd else 256

    # Keep civ_entity_mask separate for manual restoration
    civ_mask_to_set = None
    if 'civ_entity_mask' in sd:
        civ_mask_to_set = sd.pop('civ_entity_mask')

    print(f"Model architecture: d_model={d_model}, enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}, pos_len={seq_pos_rows}")

    # Build model matching inferred dimensions
    model = BuildOrderGenerator(
        vocab_size_entity=vocab_size_entity,
        vocab_size_event=vocab_size_event,
        civ_vocab_size=civ_vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_len=seq_pos_rows
    ).to(device)

    # Handle seq_pos_embed size mismatch
    if 'seq_pos_embed.weight' in sd:
        ck_rows = sd['seq_pos_embed.weight'].shape[0]
        model_rows = model.seq_pos_embed.weight.shape[0]
        if ck_rows != model_rows:
            print(f"Note: checkpoint seq_pos_embed rows={ck_rows} differ from model rows={model_rows}; copying overlap.")
            new_w = model.seq_pos_embed.weight.data.clone()
            rows_to_copy = min(ck_rows, model_rows)
            new_w[:rows_to_copy, :] = sd['seq_pos_embed.weight'][:rows_to_copy, :]
            sd['seq_pos_embed.weight'] = new_w

    # Load state dict with strict=False
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Restore civ->entity mask if present
    if civ_mask_to_set is not None:
        try:
            civ_arr = torch.as_tensor(civ_mask_to_set)
            model.set_civ_entity_mask(civ_arr.bool().to(device))
            print("Restored civ_entity_mask from checkpoint.")
        except Exception as e:
            print(f"Warning: failed to restore civ_entity_mask from checkpoint: {e}")

    # Ensure civ->entity mask exists; if not create permissive mask
    if not hasattr(model, 'civ_entity_mask') or model.civ_entity_mask is None:
        perm_mask = torch.ones((civ_vocab_size, vocab_size_entity), dtype=torch.bool)
        perm_mask[:, 0] = False
        model.set_civ_entity_mask(perm_mask.to(device))

    # Store inverted event vocab on model for use in generation masking
    model._inv_event_vocab = inv_ev
    
    inv_vocabs = {'entity': inv_ent, 'event': inv_ev}
    return model, vocabs, inv_vocabs


def evaluate_on_dataset(model, dataloader, device, vocabs):
    """
    Evaluate model on a dataset using the same methodology as training.
    Uses teacher forcing (ground truth inputs) and measures logit accuracy.
    This reproduces the training evaluation metrics.
    """
    model.eval()
    
    total_correct_entity = 0
    total_correct_top3 = 0
    total_correct_top5 = 0
    total_correct_top10 = 0
    total_samples = 0
    losses = []
    
    import torch.nn as nn
    
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Derive allowed mask
            if hasattr(model, 'civ_entity_mask') and model.civ_entity_mask is not None:
                allowed = model.civ_entity_mask[batch['player_civ']]
            else:
                allowed = None

            # Forward with teacher forcing (same as training evaluation)
            preds = model(
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ'],
                target_sequence_length=batch['entity_ids'].size(1),
                target_win_prob=None,  # Don't leak labels
                teacher_forcing_ratio=1.0,  # Full teacher forcing
                ground_truth=(batch['entity_ids'], batch['event_ids'], batch['times']),
                allowed_entities_mask=allowed
            )
            
            entity_logits, event_logits, time_preds, win_logits = preds
            
            # Compute loss
            entity_targets = batch['entity_ids']
            event_targets = batch['event_ids']
            time_targets = batch['times']
            
            entity_loss = nn.CrossEntropyLoss(ignore_index=0)(
                entity_logits.view(-1, entity_logits.size(-1)),
                entity_targets.view(-1)
            )
            event_loss = nn.CrossEntropyLoss(ignore_index=0)(
                event_logits.view(-1, event_logits.size(-1)),
                event_targets.view(-1)
            )
            time_loss = nn.MSELoss()(time_preds, time_targets)
            
            total_loss = entity_loss + event_loss + 0.8 * time_loss
            losses.append(total_loss.item())

            # Entity accuracy calculation (same as training)
            batch_size, seq_len = entity_targets.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    target_id = entity_targets[b, s].item()
                    if target_id == 0:  # skip padding
                        continue
                    
                    logits_step = entity_logits[b, s]
                    
                    # Top-1 accuracy
                    top1_pred = logits_step.argmax().item()
                    if top1_pred == target_id:
                        total_correct_entity += 1
                    
                    # Top-k accuracies
                    topk_values, topk_indices = torch.topk(logits_step, min(10, logits_step.size(0)))
                    topk_list = topk_indices.tolist()
                    
                    if target_id in topk_list[:3]:
                        total_correct_top3 += 1
                    if target_id in topk_list[:5]:
                        total_correct_top5 += 1
                    if target_id in topk_list[:10]:
                        total_correct_top10 += 1
                    
                    total_samples += 1

    avg_loss = float(np.mean(losses)) if losses else float('nan')
    entity_accuracy = total_correct_entity / total_samples if total_samples > 0 else float('nan')
    entity_top3_accuracy = total_correct_top3 / total_samples if total_samples > 0 else float('nan')
    entity_top5_accuracy = total_correct_top5 / total_samples if total_samples > 0 else float('nan')
    entity_top10_accuracy = total_correct_top10 / total_samples if total_samples > 0 else float('nan')
    
    return {
        'loss': avg_loss,
        'entity_accuracy': entity_accuracy,
        'entity_top3_accuracy': entity_top3_accuracy,
        'entity_top5_accuracy': entity_top5_accuracy,
        'entity_top10_accuracy': entity_top10_accuracy,
        'total_samples': total_samples
    }


def evaluate_autoregressive(model, dataloader, device, vocabs, max_length=None):
    """
    Evaluate model using autoregressive generation (no teacher forcing).
    This measures how well the model performs when generating from scratch,
    which is closer to actual inference behavior.
    """
    model.eval()
    
    total_correct_entity = 0
    total_correct_event = 0
    total_samples = 0
    sequence_accuracy_sum = 0.0
    num_sequences = 0
    
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            entity_targets = batch['entity_ids']
            event_targets = batch['event_ids']
            batch_size, seq_len = entity_targets.shape
            
            gen_len = max_length if max_length else seq_len
            
            # Generate autoregressively for each sample
            for b in range(batch_size):
                player_civ = batch['player_civ'][b:b+1]
                enemy_civ = batch['enemy_civ'][b:b+1]
                
                # Use forward with teacher_forcing_ratio=0 (autoregressive)
                preds = model(
                    player_civ=player_civ,
                    enemy_civ=enemy_civ,
                    target_sequence_length=min(gen_len, seq_len),
                    target_win_prob=None,
                    teacher_forcing_ratio=0.0,  # No teacher forcing
                    ground_truth=None,
                    allowed_entities_mask=None
                )
                
                entity_logits, event_logits, _, _ = preds
                
                # Get predictions
                pred_entities = entity_logits.argmax(dim=-1).squeeze(0)  # (seq_len,)
                pred_events = event_logits.argmax(dim=-1).squeeze(0)
                
                # Compare with ground truth
                target_ent = entity_targets[b, :pred_entities.size(0)]
                target_ev = event_targets[b, :pred_events.size(0)]
                
                correct_in_seq = 0
                for s in range(min(pred_entities.size(0), target_ent.size(0))):
                    target_id = target_ent[s].item()
                    if target_id == 0:  # skip padding
                        continue
                    
                    if pred_entities[s].item() == target_id:
                        total_correct_entity += 1
                        correct_in_seq += 1
                    if pred_events[s].item() == target_ev[s].item():
                        total_correct_event += 1
                    total_samples += 1
                
                # Sequence-level accuracy
                valid_targets = (target_ent != 0).sum().item()
                if valid_targets > 0:
                    sequence_accuracy_sum += correct_in_seq / valid_targets
                    num_sequences += 1

    entity_accuracy = total_correct_entity / total_samples if total_samples > 0 else float('nan')
    event_accuracy = total_correct_event / total_samples if total_samples > 0 else float('nan')
    seq_accuracy = sequence_accuracy_sum / num_sequences if num_sequences > 0 else float('nan')
    
    return {
        'entity_accuracy': entity_accuracy,
        'event_accuracy': event_accuracy,
        'avg_sequence_accuracy': seq_accuracy,
        'total_samples': total_samples,
        'num_sequences': num_sequences
    }


def pretty_print_build(sequence_entities, sequence_events, sequence_times, inv_ent, inv_ev):
    lines = []
    for i, (e, ev, t) in enumerate(zip(sequence_entities, sequence_events, sequence_times)):
        ent_name = inv_ent.get(int(e), '<UNK>')
        ev_name = inv_ev.get(int(ev), '<UNK>')
        lines.append(f"{i+1:02d}. {t:7.1f}s  {ev_name:<10} {ent_name}")
    return "\n".join(lines)


def print_step_probabilities(step_probs_list, inv_ent, inv_ev=None, top_n=5, show_times=True):
    """Print probability distributions for top entities at each generation step.
    
    Args:
        step_probs_list: List of dicts with 'probs' (tensor) and 'selected' (int) for each step
        inv_ent: Inverted entity vocabulary (id -> name)
        inv_ev: Inverted event vocabulary (id -> name), optional
        top_n: Number of top entities to display
        show_times: Whether to display time predictions
    """
    print("\n" + "=" * 70)
    print("STEP-BY-STEP PROBABILITY DISTRIBUTIONS")
    print("=" * 70)
    
    for step_idx, step_data in enumerate(step_probs_list):
        probs = step_data['probs']
        selected_id = step_data['selected']
        selected_name = inv_ent.get(selected_id, '<UNK>')
        
        # Get time info if available
        time_info = ""
        if show_times and 'time_delta' in step_data and 'cumulative_time' in step_data:
            time_delta = step_data['time_delta']
            cumulative_time = step_data['cumulative_time']
            time_info = f" | Δt={time_delta:.1f}s, T={cumulative_time:.1f}s"
        
        # Get event info if available
        event_info = ""
        if inv_ev is not None and 'selected_event' in step_data:
            event_name = inv_ev.get(step_data['selected_event'], '<UNK>')
            event_info = f" [{event_name}]"
        
        # Get top N probabilities and indices
        top_vals, top_idx = torch.topk(probs, min(top_n, probs.size(-1)))
        
        print(f"\nStep {step_idx + 1}: Selected → {selected_name}{event_info}{time_info}")
        print("-" * 50)
        
        for rank, (prob, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist())):
            ent_name = inv_ent.get(idx, '<UNK>')
            bar_len = int(prob * 40)  # Scale probability to bar length
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = " ← SELECTED" if idx == selected_id else ""
            print(f"  {rank + 1}. {ent_name:<30} {prob * 100:5.2f}% |{bar}|{marker}")
    
    print("\n" + "=" * 70 + "\n")


def generate_with_probabilities(model, player_civ, enemy_civ, max_length, device, temperature=1.0, 
                                 top_k=0, top_p=0.0, use_sampling=False):
    """Generate build order with step-wise probability tracking.
    
    Args:
        model: The BuildOrderGenerator model
        player_civ: Player civilization tensor (1,)
        enemy_civ: Enemy civilization tensor (1,)
        max_length: Maximum sequence length to generate
        device: Torch device
        temperature: Softmax temperature (higher = more diverse, lower = more focused)
        top_k: If > 0, only sample from top k most likely tokens
        top_p: If > 0, use nucleus sampling (sample from smallest set with cumulative prob >= top_p)
        use_sampling: If True, sample from distribution; if False, use argmax (greedy)
    
    Returns:
        entities: List of entity IDs
        events: List of event IDs  
        times: List of timestamps
        step_probs: List of dicts with probability info for each step
        win_prob: Final win probability
    """
    import torch.nn.functional as F
    
    model.eval()
    
    # Encode condition
    memory = model.encode_condition(player_civ, enemy_civ, None)  # (1, 1, d_model)
    
    # Initialize sequences
    entities = []
    events = []
    times = []
    step_probs = []
    
    # Start embeddings
    entity_embs = model.start_token_entity.expand(1, -1, -1).clone()  # (1, 1, d_model)
    event_embs = model.start_token_event.expand(1, -1, -1).clone()    # (1, 1, d_model)
    last_time = 0.0
    
    with torch.no_grad():
        for step in range(max_length):
            seq_len = len(entities) + 1  # +1 for start token
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
            pos_emb = model.seq_pos_embed(positions)
            
            decoder_input = model.decoder_norm(entity_embs + event_embs + pos_emb)
            tgt_mask = model.generate_square_subsequent_mask(seq_len).to(device)
            
            decoder_output = model.decoder(decoder_input, memory, tgt_mask=tgt_mask)
            last_output = decoder_output[:, -1:, :]
            
            # Get logits
            entity_logits = (model.entity_head(last_output) / temperature).squeeze(0).squeeze(0)
            event_logits = (model.event_head(last_output) / temperature).squeeze(0).squeeze(0)
            time_delta = torch.relu(model.time_head(last_output)).squeeze()
            
            # Mask out UNKNOWN and PAD events - they should never be generated
            # Index 0 is typically <PAD>, look up UNKNOWN dynamically
            event_logits[0] = -1e9  # <PAD>
            if hasattr(model, '_inv_event_vocab'):
                for ev_id, ev_name in model._inv_event_vocab.items():
                    ev_upper = ev_name.upper()
                    if 'UNK' in ev_upper or 'PAD' in ev_upper:
                        event_logits[ev_id] = -1e9
            
            # Apply civ-based mask if available
            if hasattr(model, 'civ_entity_mask') and model.civ_entity_mask is not None:
                civ_id = player_civ[0]
                allowed = model.civ_entity_mask[civ_id].to(device)
                entity_logits = entity_logits.masked_fill(~allowed, model.mask_value)
            
            # Convert to probabilities
            entity_probs = F.softmax(entity_logits, dim=-1)
            event_probs = F.softmax(event_logits, dim=-1)
            
            # Apply top-k filtering if specified
            if top_k > 0:
                # Keep only top k tokens, set rest to 0
                top_k_vals, top_k_idx = torch.topk(entity_probs, min(top_k, entity_probs.size(-1)))
                entity_probs_filtered = torch.zeros_like(entity_probs)
                entity_probs_filtered.scatter_(-1, top_k_idx, top_k_vals)
                entity_probs = entity_probs_filtered / entity_probs_filtered.sum()  # renormalize
            
            # Apply nucleus (top-p) sampling if specified
            if top_p > 0:
                sorted_probs, sorted_indices = torch.sort(entity_probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                # Find cutoff index
                sorted_mask = cumsum_probs <= top_p
                # Always keep at least one token
                sorted_mask[0] = True
                entity_probs_filtered = torch.zeros_like(entity_probs)
                entity_probs_filtered.scatter_(-1, sorted_indices[sorted_mask], sorted_probs[sorted_mask])
                entity_probs = entity_probs_filtered / entity_probs_filtered.sum()  # renormalize
            
            # Select token: sample or greedy
            if use_sampling:
                # Sample from the distribution
                selected_entity = torch.multinomial(entity_probs, 1).item()
                selected_event = torch.multinomial(event_probs, 1).item()
            else:
                # Greedy selection
                selected_entity = entity_probs.argmax().item()
                selected_event = event_probs.argmax().item()
            
            new_time = last_time + float(time_delta.item())
            
            # Store step info
            step_probs.append({
                'probs': entity_probs.cpu(),
                'selected': selected_entity,
                'event_probs': event_probs.cpu(),
                'selected_event': selected_event,
                'time_delta': float(time_delta.item()),
                'cumulative_time': new_time
            })
            
            # Update sequences
            entities.append(selected_entity)
            events.append(selected_event)
            times.append(new_time)
            last_time = new_time
            
            # Update embeddings
            new_entity_id = torch.tensor([selected_entity], device=device)
            new_event_id = torch.tensor([selected_event], device=device)
            new_entity_emb = model.entity_embed(new_entity_id).unsqueeze(0)
            new_event_emb = model.event_embed(new_event_id).unsqueeze(0)
            entity_embs = torch.cat([entity_embs, new_entity_emb], dim=1)
            event_embs = torch.cat([event_embs, new_event_emb], dim=1)
        
        # Get final win probability
        seq_len = len(entities) + 1
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        pos_emb = model.seq_pos_embed(positions)
        decoder_input = model.decoder_norm(entity_embs + event_embs + pos_emb)
        tgt_mask = model.generate_square_subsequent_mask(seq_len).to(device)
        decoder_output = model.decoder(decoder_input, memory, tgt_mask=tgt_mask)
        final_output = decoder_output[:, -1, :]
        win_prob = float(torch.sigmoid(model.win_prob_head(final_output)).squeeze().item())
    
    return entities, events, times, step_probs, win_prob


def main():
    p = argparse.ArgumentParser(description='Inference and evaluation for BuildOrderGenerator')
    p.add_argument('--ckpt', type=str, default='best_buildorder_model.pt', help='Path to checkpoint containing model_state and vocabs')
    
    # Generation mode arguments
    p.add_argument('--player_civ', type=str, default=None, help='Player civilization name (string, e.g. "English")')
    p.add_argument('--enemy_civ', type=str, default=None, help='Enemy civilization name (string)')
    p.add_argument('--beam_width', type=int, default=4, help='Beam width for beam search')
    p.add_argument('--max_length', type=int, default=12, help='Maximum build order length to generate (steps)')
    p.add_argument('--top_k', type=int, default=1, help='Print top K beams')
    p.add_argument('--greedy', type=int, default=0, help='Use greedy decoding instead of beam search')
    p.add_argument('--show_probs', type=int, default=0, help='Show probability distribution for top entities at each step (1=enabled)')
    p.add_argument('--top_probs', type=int, default=5, help='Number of top entities to show in probability distribution')
    
    # Sampling parameters to avoid mode collapse on Villager
    p.add_argument('--temperature', type=float, default=1.0, help='Temperature for softmax (higher = more diverse)')
    p.add_argument('--sample_top_k', type=int, default=0, help='If > 0, sample from top-k most likely tokens')
    p.add_argument('--sample_top_p', type=float, default=0.0, help='If > 0, use nucleus sampling (sample from tokens with cumulative prob >= top_p)')
    p.add_argument('--use_sampling', type=int, default=0, help='If 1, use sampling instead of greedy decoding')
    
    # Evaluation mode arguments
    p.add_argument('--eval_csv', type=str, default=None, help='CSV file to evaluate on (reproduces training metrics)')
    p.add_argument('--eval_split', type=float, default=0.1, help='Fraction of data to use for validation split (uses same split as training)')
    p.add_argument('--eval_mode', type=str, default='teacher_forcing', choices=['teacher_forcing', 'autoregressive', 'both'],
                   help='Evaluation mode: teacher_forcing matches training metrics, autoregressive shows real generation performance')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    p.add_argument('--max_len', type=int, default=None, help='Max sequence length for evaluation dataset')
    p.add_argument('--truncation_strategy', type=str, default='head_tail', choices=['head', 'tail', 'head_tail'])
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducible train/val split')
    
    p.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load model using helper function
    model, vocabs, inv_vocabs = load_model_from_checkpoint(args.ckpt, device)
    inv_ent = inv_vocabs['entity']
    inv_ev = inv_vocabs['event']
    civ_vocab = vocabs['civ_vocab']
    
    # =========================================================================
    # EVALUATION MODE: Evaluate on a dataset to reproduce training metrics
    # =========================================================================
    if args.eval_csv is not None:
        print(f"\n{'='*60}")
        print("EVALUATION MODE")
        print(f"{'='*60}")
        print(f"CSV: {args.eval_csv}")
        print(f"Eval mode: {args.eval_mode}")
        print(f"Seed: {args.seed}")
        print()
        
        # Import dataset utilities
        import pandas as pd
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import train_test_split
        import sys
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from aoe_player_game_datset import build_vocabs, AoEEventDataset, collate_fn
        
        # Load dataset with the same vocabs as the model
        df = pd.read_csv(args.eval_csv)
        
        # Determine max_len - IMPORTANT: limit to model's positional embedding size
        # to avoid out-of-bounds GPU memory access (HSA_STATUS_ERROR_EXCEPTION)
        grouped = df.groupby(['game_id', 'profile_id']).size()
        dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0
        model_max_len = model.seq_pos_embed.num_embeddings  # Model's position embedding limit
        desired_max_len = args.max_len if args.max_len is not None else min(dataset_longest, 1024)
        if desired_max_len > model_max_len:
            print(f"Warning: Requested max_len={desired_max_len} exceeds model's positional embedding size ({model_max_len}). Clamping to {model_max_len}.")
            desired_max_len = model_max_len
        
        dataset = AoEEventDataset(
            args.eval_csv, 
            vocabs['entity_vocab'], 
            vocabs['event_vocab'], 
            vocabs['civ_vocab'], 
            vocabs['map_vocab'], 
            max_len=desired_max_len, 
            truncation_strategy=args.truncation_strategy
        )
        
        # Split by game_id using the same seed as training
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        all_game_ids = list({ex['game_id'] for ex in dataset.examples})
        train_games, val_games = train_test_split(all_game_ids, test_size=args.eval_split, random_state=args.seed)
        val_idx = [i for i, ex in enumerate(dataset.examples) if ex['game_id'] in val_games]
        
        val_ds = Subset(dataset, val_idx)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        print(f"Validation samples: {len(val_ds)}")
        print()
        
        # Run evaluation
        if args.eval_mode in ['teacher_forcing', 'both']:
            print("--- Teacher Forcing Evaluation (matches training metrics) ---")
            tf_results = evaluate_on_dataset(model, val_loader, device, vocabs)
            print(f"  Loss: {tf_results['loss']:.4f}")
            print(f"  Entity Accuracy (Top-1): {tf_results['entity_accuracy']:.4f} ({tf_results['entity_accuracy']*100:.2f}%)")
            print(f"  Entity Accuracy (Top-3): {tf_results['entity_top3_accuracy']:.4f} ({tf_results['entity_top3_accuracy']*100:.2f}%)")
            print(f"  Entity Accuracy (Top-5): {tf_results['entity_top5_accuracy']:.4f} ({tf_results['entity_top5_accuracy']*100:.2f}%)")
            print(f"  Entity Accuracy (Top-10): {tf_results['entity_top10_accuracy']:.4f} ({tf_results['entity_top10_accuracy']*100:.2f}%)")
            print(f"  Total samples evaluated: {tf_results['total_samples']}")
            print()
        
        if args.eval_mode in ['autoregressive', 'both']:
            print("--- Autoregressive Evaluation (real generation performance) ---")
            ar_results = evaluate_autoregressive(model, val_loader, device, vocabs, max_length=args.max_length)
            print(f"  Entity Accuracy: {ar_results['entity_accuracy']:.4f} ({ar_results['entity_accuracy']*100:.2f}%)")
            print(f"  Event Accuracy: {ar_results['event_accuracy']:.4f} ({ar_results['event_accuracy']*100:.2f}%)")
            print(f"  Avg Sequence Accuracy: {ar_results['avg_sequence_accuracy']:.4f} ({ar_results['avg_sequence_accuracy']*100:.2f}%)")
            print(f"  Total samples: {ar_results['total_samples']}, Sequences: {ar_results['num_sequences']}")
            print()
        
        print(f"{'='*60}")
        print("NOTE: Teacher forcing metrics should match training validation metrics.")
        print("Autoregressive metrics show actual generation performance (typically lower).")
        print(f"{'='*60}")
        return
    
    # =========================================================================
    # GENERATION MODE: Generate build order for a civ matchup
    # =========================================================================
    if args.player_civ is None or args.enemy_civ is None:
        print("Error: --player_civ and --enemy_civ are required for generation mode.")
        print("       Use --eval_csv for evaluation mode.")
        p.print_help()
        raise SystemExit(1)

    # Map civ names to ids
    p_civ_name, p_civ_id = find_civ_id(args.player_civ, civ_vocab)
    e_civ_name, e_civ_id = find_civ_id(args.enemy_civ, civ_vocab)
    if p_civ_id is None or e_civ_id is None:
        print('Unknown civ name(s). Available civs:')
        print(', '.join(sorted(list(civ_vocab.keys())[:200])))
        raise SystemExit(1)

    p_tensor = torch.tensor([p_civ_id], dtype=torch.long, device=device)
    e_tensor = torch.tensor([e_civ_id], dtype=torch.long, device=device)

    if args.greedy == 1:
        if args.show_probs == 1:
            # Greedy decoding with probability tracking
            ents, evs, times, step_probs, best_win = generate_with_probabilities(
                model, p_tensor, e_tensor, args.max_length, device,
                temperature=args.temperature,
                top_k=args.sample_top_k,
                top_p=args.sample_top_p,
                use_sampling=bool(args.use_sampling)
            )
            sampling_mode = "sampling" if args.use_sampling else "greedy"
            print(f"{sampling_mode.capitalize()} build for {p_civ_name} vs {e_civ_name} — win_prob={best_win:.3f}")
            if args.temperature != 1.0:
                print(f"  (temperature={args.temperature})")
            if args.sample_top_k > 0:
                print(f"  (top_k={args.sample_top_k})")
            if args.sample_top_p > 0:
                print(f"  (top_p={args.sample_top_p})")
            print()
            print(pretty_print_build(ents, evs, times, inv_ent, inv_ev))
            print_step_probabilities(step_probs, inv_ent, inv_ev=inv_ev, top_n=args.top_probs, show_times=True)
        else:
            # Standard greedy decoding
            sequences, win_probs = model.beam_search_generate(
                player_civ=p_tensor, enemy_civ=e_tensor, 
                beam_width=1, max_length=args.max_length,
                temperature=args.temperature
            )
            best_seq = sequences[0]
            best_win = float(win_probs[0].item())
            print(f"Greedy build for {p_civ_name} vs {e_civ_name} — win_prob={best_win:.3f}\n")
            ents, evs, times = best_seq
            print(pretty_print_build(ents, evs, times, inv_ent, inv_ev))
    else:
        # Beam search
        sequences, win_probs = model.beam_search_generate(
            player_civ=p_tensor, enemy_civ=e_tensor, 
            beam_width=args.beam_width, max_length=args.max_length
        )
        all_seqs = sequences
        all_wins = win_probs.cpu().numpy().tolist()
        top_k = min(args.top_k, len(all_seqs))
        print(f"Top {top_k} beams for {p_civ_name} vs {e_civ_name} (beam_width={args.beam_width}, max_length={args.max_length}):\n")
        for i in range(top_k):
            ents, evs, times = all_seqs[i]
            win = all_wins[i]
            print(f"=== Beam {i+1}: win_prob={win:.4f} ===")
            print(pretty_print_build(ents, evs, times, inv_ent, inv_ev))
            print()


if __name__ == '__main__':
    main()
