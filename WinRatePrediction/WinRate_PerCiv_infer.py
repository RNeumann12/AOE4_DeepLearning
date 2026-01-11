"""
Minimal predictor that uses only player and enemy civ to estimate win probability.

Usage examples:
  python predict_civ_only.py --model best_model.pt --player_civ "chinese" --enemy_civ "english" --device cuda
  python predict_civ_only.py --model best_model.pt --player_civ "english" --enemy_civ "chinese" --device cuda

The script will:
 - load the checkpoint and vocabs
 - infer model dimensions (d_model, max_len, num_layers, dim_feedforward) from state_dict shapes where possible
 - construct an AoETransformer matching inferred dims
 - load model weights (non-strict if necessary)
 - run a forward pass with empty sequence inputs and the provided civ ids
 - print the win probability :)
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from WinRatePrediction.WinRateTransformerModel import AoETransformer
import json


def choose_nhead(d_model: int):
    # pick a divisor of d_model (try common values)
    for n in [16, 12, 8, 6, 4, 2, 1]:
        if d_model % n == 0:
            return n
    return 1


def infer_architecture_from_state(state_dict: dict):
    # d_model: from entity_embed or event_embed
    d_model = None
    if 'entity_embed.weight' in state_dict:
        d_model = state_dict['entity_embed.weight'].shape[1]
    elif 'event_embed.weight' in state_dict:
        d_model = state_dict['event_embed.weight'].shape[1]
    else:
        d_model = 128

    # max_len from seq_pos_embed
    max_len = 512
    if 'seq_pos_embed.weight' in state_dict:
        max_len = state_dict['seq_pos_embed.weight'].shape[0]

    # num_layers by counting transformer_encoder.layers.<i>
    num_layers = 3
    layer_idxs = set()
    for k in state_dict.keys():
        p = 'transformer_encoder.layers.'
        if p in k:
            try:
                rest = k.split(p)[1]
                idx = int(rest.split('.', 1)[0])
                layer_idxs.add(idx)
            except Exception:
                pass
    if len(layer_idxs) > 0:
        num_layers = max(layer_idxs) + 1

    # dim_feedforward: linear1.weight shape (dim_feedforward, d_model)
    dim_feedforward = 256
    key = 'transformer_encoder.layers.0.linear1.weight'
    if key in state_dict:
        dim_feedforward = state_dict[key].shape[0]

    return d_model, max_len, num_layers, dim_feedforward


def load_checkpoint(path: str, device: torch.device):
    chk = torch.load(path, map_location='cpu')
    model_state = chk.get('model_state', None)
    vocabs = chk.get('vocabs', None)
    if model_state is None or vocabs is None:
        raise RuntimeError('Checkpoint must contain keys "model_state" and "vocabs" (saved by train script)')
    return model_state, vocabs


def civ_to_id(civ, civ_vocab):
    print("civ_vocab:", civ_vocab)
    # Accept either a string token or an integer id
    if civ is None:
        return civ_vocab.get('<UNK>', 1)
    try:
        # string repr
        return civ_vocab.get(str(civ), civ_vocab.get('<UNK>', 1))
    except Exception:
        return civ_vocab.get('<UNK>', 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--player_civ', type=str, required=True)
    parser.add_argument('--enemy_civ', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    model_state, vocabs = load_checkpoint(args.model, device)

    # infer architecture
    d_model, max_len, num_layers, dim_feedforward = infer_architecture_from_state(model_state)
    nhead = choose_nhead(d_model)

    if args.verbose:
        print('Inferred architecture:')
        print(f'  d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_feedforward}, max_len={max_len}')

    # build model
    model = AoETransformer(
        vocab_size_entity=len(vocabs['entity_vocab']),
        vocab_size_event=len(vocabs['event_vocab']),
        civ_vocab_size=len(vocabs['civ_vocab']),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len
    )

    # load weights (allow non-strict in case some sizes differ)
    load_result = model.load_state_dict(model_state, strict=False)
    missing, unexpected = load_result.missing_keys, load_result.unexpected_keys

    if args.verbose:
        print('Loaded checkpoint (non-strict).')
        print('  missing keys count:', len(missing))
        print('  unexpected keys count:', len(unexpected))

    model.to(device)
    model.eval()

    # Map civ tokens to ids
    civ_vocab = vocabs['civ_vocab']
    p_id = civ_to_id(args.player_civ, civ_vocab)
    e_id = civ_to_id(args.enemy_civ, civ_vocab)

    # FIX: Create a dummy sequence with reasonable values instead of empty
    # The model expects some sequence to process, even if minimal
    # Use padding tokens for a minimal sequence
    entity_vocab = vocabs['entity_vocab']
    event_vocab = vocabs['event_vocab']
    
    # Get padding indices (assuming <PAD> tokens exist in vocabs)
    entity_pad_id = entity_vocab.get('<PAD>', 0)
    event_pad_id = event_vocab.get('<PAD>', 0)
    
    # Create a minimal sequence (1-3 events) that represents "early game"
    # This gives the model some structure to work with
    seq_len = 3  # Small but non-zero sequence
    
    # Create dummy sequence data
    entity_ids = torch.full((1, seq_len), entity_pad_id, dtype=torch.long, device=device)
    event_ids = torch.full((1, seq_len), event_pad_id, dtype=torch.long, device=device)
    
    # Create increasing times (0, 30, 60 seconds) to simulate early game
    times = torch.tensor([[0.0, 30.0, 60.0]], dtype=torch.float32, device=device)
    
    # All positions are valid (no padding)
    attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    
    player_civ = torch.tensor([p_id], dtype=torch.long, device=device)
    enemy_civ = torch.tensor([e_id], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(entity_ids, event_ids, times, attention_mask, player_civ=player_civ, enemy_civ=enemy_civ)
        prob = torch.sigmoid(logits).cpu().item()

    print(f'Win probability for player_civ={args.player_civ} vs enemy_civ={args.enemy_civ}: {prob:.4f}')


if __name__ == '__main__':
    main()
