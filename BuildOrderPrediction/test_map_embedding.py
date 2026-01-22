#!/usr/bin/env python3
"""
Test script to verify if map embedding is properly used in the MoE WithDecoder model.

This script generates build orders for the same matchup (player civ vs enemy civ)
on different maps and checks if the build orders are different. If map embedding
is working properly, we should see different build orders for different maps.
"""

import os
import sys
import torch
import json
from tabulate import tabulate
from collections import Counter

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.abspath(__file__))
build_order_path = os.path.join(repo_root, 'BuildOrderPrediction')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if build_order_path not in sys.path:
    sys.path.insert(0, build_order_path)

from MoE_WithDecoder_infer import (
    load_model, 
    generate_build_order, 
    invert_vocab,
    find_vocab_id
)


def test_map_embedding(
    checkpoint_path: str,
    player_civ: str,
    enemy_civ: str,
    maps_to_test: list,
    num_steps: int = 30,
    seed: int = 42,
    temperature: float = 0.3,
    greedy: bool = True,
    device: str = 'cuda'
):
    """
    Test if map embedding affects the generated build order.
    
    Args:
        checkpoint_path: Path to model checkpoint
        player_civ: Player civilization name
        enemy_civ: Enemy civilization name
        maps_to_test: List of map names to test
        num_steps: Number of build steps to generate
        seed: Random seed for reproducibility
        temperature: Sampling temperature
        greedy: Use greedy sampling for consistency
        device: Device to use (cuda or cpu)
    
    Returns:
        Dictionary with results for each map
    """
    device_obj = torch.device(device)
    
    # Load model
    print("=" * 80)
    print("TESTING MAP EMBEDDING IN MoE WithDecoder MODEL")
    print("=" * 80)
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping = load_model(
        checkpoint_path, device_obj
    )
    
    # Look up vocabulary IDs
    player_civ_id = find_vocab_id(player_civ, civ_vocab, "civilization vocabulary")
    enemy_civ_id = find_vocab_id(enemy_civ, civ_vocab, "civilization vocabulary")
    
    print(f"\nTest Configuration:")
    print(f"  Player Civ: {player_civ} (id={player_civ_id})")
    print(f"  Enemy Civ:  {enemy_civ} (id={enemy_civ_id})")
    print(f"  Maps:       {', '.join(maps_to_test)}")
    print(f"  Build Steps: {num_steps}")
    print(f"  Temperature: {temperature}")
    print(f"  Greedy:      {greedy}")
    print(f"  Seed:        {seed}")
    
    # Generate build orders for each map
    results = {}
    inv_entity = invert_vocab(entity_vocab)
    
    print("\n" + "=" * 80)
    print("GENERATING BUILD ORDERS")
    print("=" * 80)
    
    for map_name in maps_to_test:
        print(f"\nGenerating for map: {map_name}")
        
        try:
            map_id = find_vocab_id(map_name, map_vocab, "map vocabulary")
        except SystemExit:
            print(f"  Error: Map not found")
            continue
        
        # Generate build order
        entities = generate_build_order(
            model=model,
            player_civ_id=player_civ_id,
            enemy_civ_id=enemy_civ_id,
            map_id=map_id,
            entity_vocab=entity_vocab,
            num_steps=num_steps,
            device=device_obj,
            temperature=temperature,
            civ_entity_mapping=civ_entity_mapping,
            player_civ_name=player_civ,
            greedy=greedy,
            seed=seed,
            exclude_special=True,
        )
        
        # Convert entity IDs to names
        entity_names = []
        special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
        for ent_id in entities:
            entity_name = inv_entity.get(ent_id, f"<ID:{ent_id}>")
            if entity_name not in special_tokens:
                entity_names.append(entity_name)
        
        results[map_name] = {
            'entities': entities,
            'entity_names': entity_names,
            'map_id': map_id,
            'entity_counts': Counter(entity_names)
        }
        
        print(f"  Generated {len(entity_names)} steps")
        print(f"  First 10: {', '.join(entity_names[:10])}")
    
    return results, entity_vocab, inv_entity, player_civ, enemy_civ


def compare_build_orders(results: dict, player_civ: str, enemy_civ: str):
    """
    Compare build orders across different maps and analyze differences.
    """
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    
    maps = list(results.keys())
    
    # 1. Check if all build orders are identical (bad sign - map not used)
    print("\n1. EXACT MATCH ANALYSIS")
    print("-" * 80)
    
    first_map = maps[0]
    first_order = results[first_map]['entity_names']
    
    identical_count = 0
    different_count = 0
    
    comparison_table = []
    for map_name in maps:
        order = results[map_name]['entity_names']
        is_identical = order == first_order
        
        if is_identical:
            identical_count += 1
            status = "IDENTICAL ❌"
        else:
            different_count += 1
            status = "DIFFERENT ✓"
        
        # Count differences
        diff_count = 0
        for i in range(min(len(first_order), len(order))):
            if first_order[i] != order[i]:
                diff_count += 1
        
        comparison_table.append([
            map_name,
            len(order),
            diff_count,
            status
        ])
    
    print(tabulate(
        comparison_table,
        headers=["Map", "Steps", "Diffs from First", "Status"],
        tablefmt="grid"
    ))
    
    print(f"\n  Summary: {identical_count} identical, {different_count} different")
    if identical_count == len(maps):
        print("  ⚠️  WARNING: All build orders are identical - map embedding may NOT be working!")
    else:
        print("  ✓ Build orders vary across maps - map embedding appears to be working")
    
    # 2. Entity frequency comparison
    print("\n2. ENTITY FREQUENCY ANALYSIS")
    print("-" * 80)
    
    all_entities = set()
    for result in results.values():
        all_entities.update(result['entity_counts'].keys())
    
    entity_freq_table = []
    for entity in sorted(all_entities)[:20]:  # Top 20
        freq_by_map = []
        for map_name in sorted(maps):
            count = results[map_name]['entity_counts'].get(entity, 0)
            total = len(results[map_name]['entity_names'])
            pct = (count / total * 100) if total > 0 else 0
            freq_by_map.append(f"{count} ({pct:.1f}%)")
        
        entity_freq_table.append([entity] + freq_by_map)
    
    print(tabulate(
        entity_freq_table,
        headers=["Entity"] + sorted(maps),
        tablefmt="grid"
    ))
    
    # 3. Order similarity using Jaccard index
    print("\n3. ORDER SIMILARITY (First 10 steps)")
    print("-" * 80)
    
    similarity_table = []
    for i, map1 in enumerate(maps):
        row = [map1]
        for map2 in maps:
            order1 = set(results[map1]['entity_names'][:10])
            order2 = set(results[map2]['entity_names'][:10])
            
            if not order1 or not order2:
                jaccard = 0.0
            else:
                jaccard = len(order1 & order2) / len(order1 | order2)
            
            row.append(f"{jaccard:.3f}")
        similarity_table.append(row)
    
    print(tabulate(
        similarity_table,
        headers=["Map"] + maps,
        tablefmt="grid"
    ))
    
    # 4. Detailed build order comparison
    print("\n4. DETAILED BUILD ORDER COMPARISON (First 15 steps)")
    print("-" * 80)
    
    max_steps = max(len(results[m]['entity_names']) for m in maps)
    max_steps = min(max_steps, 15)
    
    for step in range(max_steps):
        step_data = [f"Step {step + 1}"]
        for map_name in sorted(maps):
            entities = results[map_name]['entity_names']
            if step < len(entities):
                step_data.append(entities[step])
            else:
                step_data.append("-")
        
        print(f"{' | '.join(step_data)}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test if map embedding is working properly in MoE WithDecoder model'
    )
    parser.add_argument('--checkpoint', type=str, default='BuildOrderPrediction/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--player_civ', type=str, default='English',
                       help='Player civilization')
    parser.add_argument('--enemy_civ', type=str, default='French',
                       help='Enemy civilization')
    parser.add_argument('--maps', type=str, default='Altai,Boulder Bay,Canal,Carmel,Cliffside',
                       help='Comma-separated list of maps to test')
    parser.add_argument('--build_steps', type=int, default=30,
                       help='Number of build steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Sampling temperature')
    parser.add_argument('--greedy', action='store_true', default=True,
                       help='Use greedy sampling')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Parse maps
    maps_to_test = [m.strip() for m in args.maps.split(',')]
    
    # Run test
    results, entity_vocab, inv_entity, player_civ, enemy_civ = test_map_embedding(
        checkpoint_path=args.checkpoint,
        player_civ=args.player_civ,
        enemy_civ=args.enemy_civ,
        maps_to_test=maps_to_test,
        num_steps=args.build_steps,
        seed=args.seed,
        temperature=args.temperature,
        greedy=args.greedy,
        device=args.device
    )
    
    # Compare results
    compare_build_orders(results, player_civ, enemy_civ)
    
    # Export results
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    export_data = {}
    for map_name, result in results.items():
        export_data[map_name] = {
            'entity_names': result['entity_names'],
            'entity_counts': dict(result['entity_counts'])
        }
    
    with open('map_embedding_test_results.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("\n✓ Results exported to: map_embedding_test_results.json")


if __name__ == '__main__':
    main()
