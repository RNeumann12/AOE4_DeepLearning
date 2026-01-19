#!/usr/bin/env python3
"""Infer a build order using a trained SequencePredictor model.

Example:
  python BuildOrderPrediction/SimpleInfer.py --checkpoint best_model.pth --top_probs 5 --player_civ English --enemy_civ French --map "High View" --build_steps 30 --temperature 0.3
"""
import os
import sys
import argparse
from typing import Dict

import torch

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from SimpleBuildOrderPrediction_train import SequencePredictor


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Invert a vocabulary dictionary from token->id to id->token."""
    return {v: k for k, v in vocab.items()}


def find_vocab_id(name: str, vocab: Dict[str, int], vocab_name: str = "vocab") -> int:
    """Find vocabulary ID for a given name (case-insensitive)."""
    # Try exact match first
    if name in vocab:
        return vocab[name]
    
    # Try case-insensitive match
    name_lower = name.lower()
    for key, val in vocab.items():
        if key.lower() == name_lower:
            return val
    
    # List available options
    available = [k for k in vocab.keys() if k not in ('<PAD>', '<UNK>')]
    print(f"\nError: '{name}' not found in {vocab_name}.")
    print(f"Available options: {', '.join(sorted(available)[:20])}")
    if len(available) > 20:
        print(f"  ... and {len(available) - 20} more")
    sys.exit(1)


def load_model(checkpoint_path: str, device: torch.device):
    """Load model and vocabularies from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract vocabularies
    entity_vocab = checkpoint['entity_vocab']
    civ_vocab = checkpoint['civ_vocab']
    map_vocab = checkpoint['map_vocab']
    args = checkpoint.get('args', {})
    
    # Extract civ-entity mapping (if available)
    civ_entity_mapping = checkpoint.get('civ_entity_mapping', None)
    if civ_entity_mapping is not None:
        # Convert lists back to sets
        civ_entity_mapping = {k: set(v) for k, v in civ_entity_mapping.items()}
        print(f"  Civ-entity mapping: {len(civ_entity_mapping)} civilizations")
    else:
        print("  Warning: No civ-entity mapping in checkpoint (older model)")
    
    print(f"  Entity vocab size: {len(entity_vocab)}")
    print(f"  Civ vocab size: {len(civ_vocab)}")
    print(f"  Map vocab size: {len(map_vocab)}")
    
    # Infer model dimensions from state dict
    state_dict = checkpoint['model_state_dict']
    d_model = state_dict['entity_embed.weight'].shape[1]
    
    # Get model config from args or use defaults
    model = SequencePredictor(
        vocab_size_entity=len(entity_vocab),
        civ_vocab_size=len(civ_vocab),
        map_vocab_size=len(map_vocab),
        d_model=args.get('d_model', d_model),
        nhead=args.get('nhead', 4),
        num_layers=args.get('num_layers', 3),
        dropout=0.0,  # No dropout for inference
        max_seq_len=args.get('max_seq_len', 256)
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping


def generate_build_order(
    model: SequencePredictor,
    player_civ_id: int,
    enemy_civ_id: int,
    map_id: int,
    entity_vocab: Dict[str, int],
    num_steps: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = None,
    exclude_special: bool = True,
    civ_entity_mapping: Dict[str, set] = None,
    player_civ_name: str = None,
    top_probs: int = 0,
    greedy: bool = False
):
    """Generate a build order given condition parameters.
    
    Args:
        civ_entity_mapping: Dict mapping civ name -> set of valid entity names
        player_civ_name: Name of player's civilization for entity masking
        exclude_special: If True, mask out PAD and UNK tokens from predictions
        top_probs: Number of top entity probabilities to print per step (0 = disabled)
        greedy: If True, use argmax for deterministic sampling instead of multinomial
    """
    model.eval()
    
    # Prepare condition tensors
    player_civ = torch.tensor([player_civ_id], dtype=torch.long, device=device)
    enemy_civ = torch.tensor([enemy_civ_id], dtype=torch.long, device=device)
    map_tensor = torch.tensor([map_id], dtype=torch.long, device=device)
    
    # Create valid entity mask if civ_entity_mapping is available
    valid_entity_mask = None
    if civ_entity_mapping is not None and player_civ_name is not None:
        # Case-insensitive lookup for civ name
        valid_entities = set()
        player_civ_lower = player_civ_name.lower()
        for civ_key, entities in civ_entity_mapping.items():
            if civ_key.lower() == player_civ_lower:
                valid_entities = entities
                break
        if valid_entities:
            vocab_size = len(entity_vocab)
            valid_entity_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            # Always allow PAD and UNK (they get filtered separately)
            valid_entity_mask[0] = True
            valid_entity_mask[1] = True
            for entity_name in valid_entities:
                if entity_name in entity_vocab:
                    valid_entity_mask[entity_vocab[entity_name]] = True
            num_valid = valid_entity_mask.sum().item() - 2  # Exclude PAD/UNK
            print(f"  Entity mask: {num_valid} valid entities for {player_civ_name}")
    
    # Start with <BOS> token to signal game start
    # The model will predict the first real entity (usually Villager)
    bos_token_id = entity_vocab.get('<BOS>', 2)  # Default to 2 if not found
    
    print(f"  Seed: <BOS> (id={bos_token_id})")
    
    entity_seq = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    
    generated_entities = [bos_token_id]
    
    # PAD, UNK, and BOS token IDs to exclude from generation
    special_ids = {0, 1, 2}  # PAD=0, UNK=1, BOS=2
    
    # Invert vocab for printing if needed
    inv_entity = None
    if top_probs > 0:
        inv_entity = {v: k for k, v in entity_vocab.items()}
    
    with torch.no_grad():
        for step_num in range(num_steps):
            # Forward pass
            entity_logits = model(
                entity_seq,
                player_civ, enemy_civ, map_tensor,
                predict_next=True
            )
            
            # Apply civ-entity mask (most important - restricts to valid entities)
            if valid_entity_mask is not None:
                entity_logits[0, ~valid_entity_mask] = float('-inf')
            
            # Mask out special tokens if requested
            if exclude_special:
                for sid in special_ids:
                    if sid < entity_logits.shape[-1]:
                        entity_logits[0, sid] = float('-inf')
            
            # Apply temperature
            entity_logits = entity_logits / temperature
            
            # Apply top-k if specified
            if top_k is not None:
                if top_k < entity_logits.shape[-1]:
                    values, _ = torch.topk(entity_logits, top_k)
                    min_val = values[0, -1]
                    entity_logits = torch.where(
                        entity_logits < min_val,
                        torch.full_like(entity_logits, float('-inf')),
                        entity_logits
                    )
            
            # Sample
            entity_probs = torch.softmax(entity_logits, dim=-1)
            
            # Print top X entity probabilities if requested
            if top_probs > 0 and inv_entity is not None:
                # Get top probabilities and their indices
                top_vals, top_idx = torch.topk(entity_probs[0], min(top_probs, entity_probs.shape[-1]))
                print(f"\n  Step {step_num + 1} - Top {top_probs} entity probabilities:")
                for i, (prob, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist())):
                    entity_name = inv_entity.get(idx, f'<ID:{idx}>')
                    print(f"    {i+1}. {entity_name}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Sample or greedy select
            if greedy:
                next_entity = torch.argmax(entity_probs[0]).item()
            else:
                next_entity = torch.multinomial(entity_probs[0], num_samples=1).item()
            
            generated_entities.append(next_entity)
            
            # Update sequence
            entity_seq = torch.cat([
                entity_seq,
                torch.tensor([[next_entity]], device=device)
            ], dim=1)
    
    # Skip the seed token in output
    return generated_entities[1:]


def pretty_print_build_order(
    entities: list,
    inv_entity: Dict[int, str]
):
    """Print the build order in a readable format."""
    print("\n" + "=" * 60)
    print("GENERATED BUILD ORDER")
    print("=" * 60)
    print(f"{'Step':<6} {'Entity':<40}")
    print("-" * 60)
    
    step_count = 0
    special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
    
    for ent_id in entities:
        entity = inv_entity.get(ent_id, f"<ID:{ent_id}>")
        
        # Skip special tokens
        if entity in special_tokens:
            continue
        
        step_count += 1
        print(f"{step_count:<6} {entity:<40}")
    
    if step_count == 0:
        print("(No valid build steps generated)")
        print("\nNote: The model may need more training, or try:")
        print("  - Lower temperature (--temperature 0.5)")
        print("  - Different seed tokens")
    
    print("=" * 60)
    print(f"\nTotal valid steps: {step_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate build orders using trained SequencePredictor'
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--player_civ', type=str, required=True,
                       help='Player civilization name')
    parser.add_argument('--enemy_civ', type=str, required=True,
                       help='Enemy civilization name')
    parser.add_argument('--map', type=str, required=True,
                       help='Map name')
    
    # Build order length
    parser.add_argument('--build_steps', type=int, default=20,
                       help='Number of build order steps to generate (default: 20)')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (default: 1.0, higher = more random)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling (default: None = no filtering)')
    parser.add_argument('--top_probs', type=int, default=0,
                       help='Print top X entity probabilities per generation step (default: 0 = disabled)')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding (always pick highest probability) instead of sampling')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda (default: auto)')
    
    # Utility
    parser.add_argument('--list_civs', action='store_true',
                       help='List available civilizations and exit')
    parser.add_argument('--list_maps', action='store_true',
                       help='List available maps and exit')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping = load_model(
        args.checkpoint, device
    )
    
    # Handle list commands
    if args.list_civs:
        civs = [k for k in civ_vocab.keys() if k not in ('<PAD>', '<UNK>')]
        print("\nAvailable civilizations:")
        for civ in sorted(civs):
            print(f"  - {civ}")
        return
    
    if args.list_maps:
        maps = [k for k in map_vocab.keys() if k not in ('<PAD>', '<UNK>')]
        print("\nAvailable maps:")
        for m in sorted(maps):
            print(f"  - {m}")
        return
    
    # Look up vocabulary IDs
    player_civ_id = find_vocab_id(args.player_civ, civ_vocab, "civilization vocabulary")
    enemy_civ_id = find_vocab_id(args.enemy_civ, civ_vocab, "civilization vocabulary")
    map_id = find_vocab_id(args.map, map_vocab, "map vocabulary")
    
    print(f"\nGenerating build order for:")
    print(f"  Player Civ: {args.player_civ} (id={player_civ_id})")
    print(f"  Enemy Civ:  {args.enemy_civ} (id={enemy_civ_id})")
    print(f"  Map:        {args.map} (id={map_id})")
    print(f"  Steps:      {args.build_steps}")
    
    # Generate build order (with civ-entity masking if available)
    entities = generate_build_order(
        model=model,
        player_civ_id=player_civ_id,
        enemy_civ_id=enemy_civ_id,
        map_id=map_id,
        entity_vocab=entity_vocab,
        num_steps=args.build_steps,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        civ_entity_mapping=civ_entity_mapping,
        player_civ_name=args.player_civ,
        top_probs=args.top_probs,
        greedy=args.greedy
    )
    
    # Print results
    inv_entity = invert_vocab(entity_vocab)
    
    pretty_print_build_order(entities, inv_entity)


if __name__ == '__main__':
    main()
