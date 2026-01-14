#!/usr/bin/env python3
"""Infer a build order for a Civilization matchup using a trained BuildOrderGenerator.

Example:
  python infer_buildorder.py --ckpt best_buildorder_model.pt --player_civ English --enemy_civ French --beam_width 4 --max_length 12
"""
import argparse
import torch
import numpy as np
from typing import Dict

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
    inv = {v: k for k, v in vocab.items()}
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


def pretty_print_build(sequence_entities, sequence_events, sequence_times, inv_ent, inv_ev):
    lines = []
    for i, (e, ev, t) in enumerate(zip(sequence_entities, sequence_events, sequence_times)):
        ent_name = inv_ent.get(int(e), '<UNK>')
        ev_name = inv_ev.get(int(ev), '<UNK>')
        lines.append(f"{i+1:02d}. {t:7.1f}s  {ev_name:<10} {ent_name}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='best_buildorder_model.pt', help='Path to checkpoint containing model_state and vocabs')
    p.add_argument('--player_civ', type=str, required=True, help='Player civilization name (string, e.g. "English")')
    p.add_argument('--enemy_civ', type=str, required=True, help='Enemy civilization name (string)')
    p.add_argument('--beam_width', type=int, default=4, help='Beam width for beam search')
    p.add_argument('--max_length', type=int, default=12, help='Maximum build order length to generate (steps)')
    p.add_argument('--top_k', type=int, default=1, help='Print top K beams')
    p.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    p.add_argument('--greedy',type=int, default=0, help='Use greedy decoding instead of beam search')

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
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
    import re
    enc_idxs = [int(m.group(1)) for k in sd.keys() for m in [re.match(r'encoder\.layers\.(\d+)\.', k)] if m]
    dec_idxs = [int(m.group(1)) for k in sd.keys() for m in [re.match(r'decoder\.layers\.(\d+)\.', k)] if m]
    num_encoder_layers = max(enc_idxs) + 1 if enc_idxs else 1
    num_decoder_layers = max(dec_idxs) + 1 if dec_idxs else 1

    seq_pos_rows = sd['seq_pos_embed.weight'].shape[0] if 'seq_pos_embed.weight' in sd else 256

    # Remove civ_entity_mask from state dict if present (we'll restore it manually) to avoid load errors
    civ_mask_to_set = None
    if 'civ_entity_mask' in sd:
        civ_mask_to_set = sd.pop('civ_entity_mask')

    print(f"Inferring model: d_model={d_model}, enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}, pos_len={seq_pos_rows}")

    # Build model matching inferred dimensions (use checkpoint's positional embedding size)
    model = BuildOrderGenerator(
        vocab_size_entity=vocab_size_entity,
        vocab_size_event=vocab_size_event,
        civ_vocab_size=civ_vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_len=seq_pos_rows
    ).to(device)

    # Try to load state dict leniently; handle partial loads and report mismatches
    missing_keys = []
    unexpected_keys = []
    # If seq_pos_embed exists in checkpoint and has more rows than model, trim; if fewer, we'll keep as is
    if 'seq_pos_embed.weight' in sd:
        ck_rows = sd['seq_pos_embed.weight'].shape[0]
        model_rows = model.seq_pos_embed.weight.shape[0]
        if ck_rows != model_rows:
            print(f"Note: checkpoint seq_pos_embed rows={ck_rows} differ from model rows={model_rows}; copying overlap.")
            # Create a new tensor matching model shape and copy overlap
            new_w = model.seq_pos_embed.weight.data.clone()
            rows_to_copy = min(ck_rows, model_rows)
            new_w[:rows_to_copy, :] = sd['seq_pos_embed.weight'][:rows_to_copy, :]
            sd['seq_pos_embed.weight'] = new_w

    # Load state dict with strict=False to allow missing/unexpected params
    load_res = model.load_state_dict(sd, strict=False)
    # PyTorch returns a named tuple in newer versions, but to be safe compute diffs
    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    missing_keys = sorted([k for k in model_keys if k not in sd_keys])
    unexpected_keys = sorted([k for k in sd_keys if k not in model_keys])

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (these params were not restored): {missing_keys[:10]}{(' ...' if len(missing_keys)>10 else '')}")
    if unexpected_keys:
        print(f"Info: Unexpected keys in checkpoint (ignored): {unexpected_keys[:10]}{(' ...' if len(unexpected_keys)>10 else '')}")

    model.eval()

    # Restore civ->entity mask if it was present in checkpoint
    if civ_mask_to_set is not None:
        try:
            # civ_mask_to_set might be a numpy array or tensor in CPU; convert and set
            civ_arr = torch.tensor(civ_mask_to_set)
            model.set_civ_entity_mask(civ_arr.bool().to(device))
            print("Restored civ_entity_mask from checkpoint.")
        except Exception as e:
            print(f"Warning: failed to restore civ_entity_mask from checkpoint: {e}")

    # Ensure civ->entity mask exists in model; if not create permissive mask (disallow pad idx 0)
    if not hasattr(model, 'civ_entity_mask') or model.civ_entity_mask is None:
        perm_mask = torch.ones((civ_vocab_size, vocab_size_entity), dtype=torch.bool)
        perm_mask[:, 0] = False
        model.set_civ_entity_mask(perm_mask.to(device))

    # Map civ names to ids
    p_civ_name, p_civ_id = find_civ_id(args.player_civ, civ_vocab)
    e_civ_name, e_civ_id = find_civ_id(args.enemy_civ, civ_vocab)
    if p_civ_id is None or e_civ_id is None:
        print('Unknown civ name(s). Available civs (example):')
        print(', '.join(sorted(list(civ_vocab.keys())[:200])))
        raise SystemExit(1)

    p_tensor = torch.tensor([p_civ_id], dtype=torch.long, device=device)
    e_tensor = torch.tensor([e_civ_id], dtype=torch.long, device=device)

    if args.greedy == 1:
        # Greedy: set beam_width 1 and call beam search or use forward() and argmax
        sequences, win_probs = model.beam_search_generate(player_civ=p_tensor, enemy_civ=e_tensor, beam_width=1, max_length=args.max_length)
        # sequences is list of (entities, events, times) for each batch item. Since batch size 1, take sequences[0] which is tuple
        best_seq = sequences[0]
        best_win = float(win_probs[0].item())
        print(f"Greedy build for {p_civ_name} vs {e_civ_name} — win_prob={best_win:.3f}\n")
        ents, evs, times = best_seq
        print(pretty_print_build(ents, evs, times, inv_ent, inv_ev))
    else:
        sequences, win_probs = model.beam_search_generate(player_civ=p_tensor, enemy_civ=e_tensor, beam_width=args.beam_width, max_length=args.max_length)
        # For batch size 1
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
