#!/usr/bin/env python3
"""Infer a build order using a trained MoE WithDecoder SequencePredictor model.

This version uses the encoder-decoder architecture where:
- Encoder: Processes conditions (player civ, enemy civ, map) into memory
- Decoder: Uses cross-attention to encoder memory for generation

Example:
  python BuildOrderPrediction/MoE_WithDecoder_infer.py --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth --top_probs 5 --player_civ English --enemy_civ French --map "Dry Arabia" --build_steps 30 --temperature 0.3 --top_p 0.9 --seed 42
  python BuildOrderPrediction/MoE_WithDecoder_infer.py --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth --player_civ sengoku_daimyo --enemy_civ French --map "Altai" --build_steps 30 --greedy
  python BuildOrderPrediction/MoE_WithDecoder_infer.py --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth --player_civ English --enemy_civ French --map "Four Lakes" --build_steps 30 --greedy
  python BuildOrderPrediction/MoE_WithDecoder_infer.py --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth --player_civ French --enemy_civ French --map "Four Lakes" --build_steps 30 --greedy
  python BuildOrderPrediction/MoE_WithDecoder_infer.py --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth --player_civ abbasid_dynasty --enemy_civ English --map "Dry Arabia" --build_steps 30 --greedy
  python BuildOrderPrediction/MoE_WithDecoder_infer.py --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth --player_civ English --enemy_civ French --map "Dry Arabia" --build_steps 30 --greedy

    """
import os
import sys
import argparse
import random
from typing import Dict, Optional

import torch

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from MoE_WithDecoder_train import SequencePredictor


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
    """Load MoE WithDecoder model and vocabularies from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract vocabularies
    entity_vocab = checkpoint['entity_vocab']
    civ_vocab = checkpoint['civ_vocab']
    map_vocab = checkpoint['map_vocab']
    args = checkpoint.get('args', {})

    print("civ_vocab:", civ_vocab)
    print("map_vocab:", map_vocab)

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
    
    # Check architecture type
    architecture = checkpoint.get('architecture', 'SequencePredictor_WithDecoder')
    print(f"  Architecture: {architecture}")
    
    # Get model config from args or use defaults
    model = SequencePredictor(
        vocab_size_entity=len(entity_vocab),
        civ_vocab_size=len(civ_vocab),
        map_vocab_size=len(map_vocab),
        d_model=args.get('d_model', d_model),
        nhead=args.get('nhead', 8),
        num_encoder_layers=args.get('num_encoder_layers', 4),
        num_decoder_layers=args.get('num_decoder_layers', 6),
        dim_feedforward=args.get('dim_feedforward', d_model * 4),
        dropout=0.0,  # No dropout for inference
        max_seq_len=args.get('max_seq_len', 256),
        num_experts=args.get('num_experts', 4),
        use_moe=args.get('use_moe', True),
        use_ngram=args.get('use_ngram', True),
        use_rope=args.get('use_rope', True)
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  MoE enabled: {model.use_moe}")
    print(f"  N-gram enabled: {model.use_ngram}")
    print(f"  RoPE enabled: {model.use_rope}")
    
    return model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping


def generate_build_order(
    model: SequencePredictor,
    player_civ_id: int,
    enemy_civ_id: int,
    map_id: int,
    entity_vocab: Dict[str, int],
    num_steps: int,
    device: torch.device,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: float = 0.9,
    exclude_special: bool = True,
    civ_entity_mapping: Dict[str, set] = None,
    player_civ_name: str = None,
    top_probs: int = 0,
    greedy: bool = False,
    seed: int = None
):
    """Generate a build order given condition parameters using encoder-decoder architecture.
    
    Args:
        model: Trained SequencePredictor model (with decoder)
        player_civ_id: Player civilization vocabulary ID
        enemy_civ_id: Enemy civilization vocabulary ID
        map_id: Map vocabulary ID
        entity_vocab: Entity vocabulary dictionary
        num_steps: Number of build steps to generate
        device: Torch device
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling threshold (None = disabled)
        top_p: Nucleus sampling threshold (default 0.9, set to 1.0 to disable)
        exclude_special: If True, mask out PAD and UNK tokens from predictions
        civ_entity_mapping: Dict mapping civ name -> set of valid entity names
        player_civ_name: Name of player's civilization for entity masking
        top_probs: Number of top entity probabilities to print per step (0 = disabled)
        greedy: If True, use argmax for deterministic sampling instead of multinomial
        seed: Random seed for reproducible generation. If None, results vary each run.
    
    Returns:
        List of generated entity IDs
    """
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
    
    model.eval()
    
    # Get the model's max sequence length
    max_entity_seq_len = model.max_seq_len - 1  # Leave room for safety
    
    if num_steps > max_entity_seq_len:
        print(f"  Warning: Requesting {num_steps} steps but model max context is {max_entity_seq_len}")
        print(f"           Older context will be truncated during generation.")
    
    # Prepare condition tensors
    player_civ = torch.tensor([player_civ_id], dtype=torch.long, device=device)
    enemy_civ = torch.tensor([enemy_civ_id], dtype=torch.long, device=device)
    map_tensor = torch.tensor([map_id], dtype=torch.long, device=device)
    
    # Pre-compute encoder memory (key difference from encoder-only model!)
    # This avoids re-encoding conditions at every generation step
    with torch.no_grad():
        encoder_memory, map_emb = model.encode(player_civ, enemy_civ, map_tensor)  # (1, 1, d_model) each
        print(f"  Encoder memory shape: {encoder_memory.shape}, Map emb shape: {map_emb.shape}")
    
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
    bos_token_id = entity_vocab.get('<BOS>', 2)
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
            # Truncate sequence if it exceeds model's max length
            if entity_seq.size(1) > max_entity_seq_len:
                entity_seq = entity_seq[:, -max_entity_seq_len:]
            
            # Forward pass with pre-computed encoder memory
            # This is the key difference: we pass encoder_memory to avoid re-encoding
            entity_logits = model(
                entity_seq,
                player_civ, enemy_civ, map_tensor,
                encoder_memory=encoder_memory,  # Pre-computed encoder output
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
            
            # Apply top-p (nucleus) sampling if specified
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(entity_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                entity_logits[indices_to_remove] = float('-inf')
            
            # Sample
            entity_probs = torch.softmax(entity_logits, dim=-1)
            
            # Print top X entity probabilities if requested
            if top_probs > 0 and inv_entity is not None:
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
    
    # Compute plausibility score by re-evaluating the full sequence
    step_probs = []
    with torch.no_grad():
        for i in range(1, len(generated_entities)):
            # Get prefix up to step i
            prefix_seq = torch.tensor([generated_entities[:i]], dtype=torch.long, device=device)
            
            # Truncate if needed
            if prefix_seq.size(1) > max_entity_seq_len:
                prefix_seq = prefix_seq[:, -max_entity_seq_len:]
            
            # Get logits for next token prediction
            logits = model(
                prefix_seq,
                player_civ, enemy_civ, map_tensor,
                encoder_memory=encoder_memory,
                predict_next=True
            )
            
            probs = torch.softmax(logits[0], dim=-1)
            actual_next = generated_entities[i]
            step_prob = probs[actual_next].item()
            step_probs.append(step_prob)
    
    # Calculate plausibility metrics
    import math
    if step_probs:
        log_probs = [math.log(p + 1e-10) for p in step_probs]
        mean_log_prob = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-mean_log_prob)
        geometric_mean = math.exp(mean_log_prob)
        min_prob = min(step_probs)
        max_prob = max(step_probs)
        median_prob = sorted(step_probs)[len(step_probs) // 2]
        
        plausibility_info = {
            'step_probs': step_probs,
            'geometric_mean': geometric_mean,
            'perplexity': perplexity,
            'mean_log_prob': mean_log_prob,
            'min_prob': min_prob,
            'max_prob': max_prob,
            'median_prob': median_prob
        }
    else:
        plausibility_info = None
    
    # Skip the seed token in output
    return generated_entities[1:], plausibility_info


def pretty_print_build_order(
    entities: list,
    inv_entity: Dict[int, str]
):
    """Print the build order in a readable format."""
    print("\n" + "=" * 60)
    print("GENERATED BUILD ORDER (MoE WithDecoder Model)")
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


def analyze_build_order(entities: list, inv_entity: Dict[int, str]):
    """Analyze the generated build order for common patterns."""
    special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
    
    # Count entity types
    entity_counts = {}
    for ent_id in entities:
        entity = inv_entity.get(ent_id, f"<ID:{ent_id}>")
        if entity not in special_tokens:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
    
    if not entity_counts:
        return
    
    print("\n" + "=" * 60)
    print("BUILD ORDER ANALYSIS")
    print("=" * 60)
    
    # Sort by count
    sorted_entities = sorted(entity_counts.items(), key=lambda x: -x[1])
    
    print(f"\n{'Entity':<35} {'Count':<10} {'%':<10}")
    print("-" * 55)
    total = sum(entity_counts.values())
    for entity, count in sorted_entities[:15]:
        pct = count / total * 100
        print(f"{entity:<35} {count:<10} {pct:.1f}%")
    
    if len(sorted_entities) > 15:
        print(f"  ... and {len(sorted_entities) - 15} more unique entities")
    
    # Detect age-ups
    age_markers = [e for e in entity_counts.keys() if 'Age Display' in e or 'age' in e.lower()]
    if age_markers:
        print(f"\nAge transitions detected: {', '.join(age_markers)}")
    
    print("=" * 60)


def print_plausibility_score(plausibility_info: dict, inv_entity: Dict[int, str] = None, entities: list = None):
    """Print the overall plausibility score for the generated build order."""
    if plausibility_info is None:
        print("\n(No plausibility data available)")
        return
    
    print("\n" + "=" * 60)
    print("PLAUSIBILITY SCORE")
    print("=" * 60)
    
    gm = plausibility_info['geometric_mean']
    perp = plausibility_info['perplexity']
    
    # Convert geometric mean to a 0-100 confidence score
    # Higher geometric mean = higher confidence
    confidence_pct = gm * 100
    
    print(f"\n  Overall Confidence:    {confidence_pct:.2f}%")
    print(f"  Geometric Mean Prob:   {gm:.4f}")
    print(f"  Perplexity:            {perp:.2f} (lower = more confident)")
    print(f"  Mean Log Probability:  {plausibility_info['mean_log_prob']:.4f}")
    print(f"\n  Per-step statistics:")
    print(f"    Min probability:     {plausibility_info['min_prob']:.4f} ({plausibility_info['min_prob']*100:.2f}%)")
    print(f"    Max probability:     {plausibility_info['max_prob']:.4f} ({plausibility_info['max_prob']*100:.2f}%)")
    print(f"    Median probability:  {plausibility_info['median_prob']:.4f} ({plausibility_info['median_prob']*100:.2f}%)")
    
    # Provide interpretation
    print(f"\n  Interpretation:")
    if confidence_pct >= 50:
        print(f"    ✓ High confidence - Model is very sure about this build order")
    elif confidence_pct >= 25:
        print(f"    ~ Moderate confidence - Build order follows common patterns")
    elif confidence_pct >= 10:
        print(f"    ! Low confidence - Some unconventional choices")
    else:
        print(f"    ✗ Very low confidence - Build order may be unusual or model is uncertain")
    
    # Flag any very low probability steps
    low_prob_steps = [(i, p) for i, p in enumerate(plausibility_info['step_probs']) if p < 0.05]
    if low_prob_steps and len(low_prob_steps) <= 5:
        print(f"\n  Low-confidence steps (prob < 5%):")
        for step_idx, prob in low_prob_steps:
            entity_name = ""
            if inv_entity is not None and entities is not None and step_idx < len(entities):
                entity_name = f" - {inv_entity.get(entities[step_idx], '?')}"
            print(f"    Step {step_idx + 1}: {prob*100:.2f}%{entity_name}")
    elif low_prob_steps:
        print(f"\n  {len(low_prob_steps)} steps have probability < 5%")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate build orders using trained MoE WithDecoder SequencePredictor'
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--player_civ', type=str, required=True,
                       help='Player civilization name')
    parser.add_argument('--enemy_civ', type=str, required=True,
                       help='Enemy civilization name')
    parser.add_argument('--map', type=str, default='',
                       help='Map name (optional, leave empty for no map)')
    
    # Build order length
    parser.add_argument('--build_steps', type=int, default=30,
                       help='Number of build order steps to generate (default: 30)')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Sampling temperature (default: 0.5, higher = more random, lower = more focused)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling (default: None = no filtering)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus (top-p) sampling threshold (default: 0.9, set to 1.0 to disable)')
    parser.add_argument('--top_probs', type=int, default=0,
                       help='Print top X entity probabilities per generation step (default: 0 = disabled)')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding (always pick highest probability) instead of sampling')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible generation (default: None = random each run)')
    
    # Analysis
    parser.add_argument('--analyze', action='store_true',
                       help='Print analysis of generated build order')
    
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
    # Handle empty map - use PAD token (id=0) when no map is specified
    if args.map:
        map_id = find_vocab_id(args.map, map_vocab, "map vocabulary")
    else:
        map_id = 0  # PAD token for empty/unknown map
    
    print(f"\nGenerating build order for:")
    print(f"  Player Civ: {args.player_civ} (id={player_civ_id})")
    print(f"  Enemy Civ:  {args.enemy_civ} (id={enemy_civ_id})")
    print(f"  Map:        {args.map if args.map else '(none)'} (id={map_id})")
    print(f"  Steps:      {args.build_steps}")
    print(f"  Temperature: {args.temperature}")
    if args.greedy:
        print(f"  Mode:       Greedy (deterministic)")
    else:
        print(f"  Mode:       Sampling (top_p={args.top_p})")
    
    # Set global seed if provided
    if args.seed is not None:
        print(f"  Seed:       {args.seed}")
    
    # Generate build order
    entities, plausibility_info = generate_build_order(
        model=model,
        player_civ_id=player_civ_id,
        enemy_civ_id=enemy_civ_id,
        map_id=map_id,
        entity_vocab=entity_vocab,
        num_steps=args.build_steps,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        civ_entity_mapping=civ_entity_mapping,
        player_civ_name=args.player_civ,
        top_probs=args.top_probs,
        greedy=args.greedy,
        seed=args.seed
    )
    
    # Print results
    inv_entity = invert_vocab(entity_vocab)
    pretty_print_build_order(entities, inv_entity)
    
    # Print plausibility score
    print_plausibility_score(plausibility_info, inv_entity, entities)
    
    # Optional analysis
    if args.analyze:
        analyze_build_order(entities, inv_entity)


if __name__ == '__main__':
    main()
