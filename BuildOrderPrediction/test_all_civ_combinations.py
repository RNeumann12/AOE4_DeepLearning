#!/usr/bin/env python3
"""
Test script to verify if enemy civilization affects build order generation.
Tests all possible Civ A vs Civ B combinations.
"""

import os
import sys
import torch
from collections import defaultdict
from itertools import product

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from BuildOrderPrediction.MoE_WithDecoder_infer import (
    load_model, 
    generate_build_order, 
    invert_vocab,
)


def get_build_order_signature(entities: list, inv_entity: dict, num_steps: int = 10) -> tuple:
    """Get a hashable signature of the first N build steps."""
    special_tokens = {'<PAD>', '<UNK>', '<BOS>'}
    steps = []
    for ent_id in entities:
        entity_name = inv_entity.get(ent_id, f"<ID:{ent_id}>")
        if entity_name not in special_tokens:
            steps.append(entity_name)
            if len(steps) >= num_steps:
                break
    return tuple(steps)


def test_all_civ_combinations():
    """Test all Civ A vs Civ B combinations to see if enemy civ matters."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(repo_root, 'best_model.pth')
    
    print("=" * 100)
    print("TESTING ALL CIV VS CIV COMBINATIONS - ENEMY CIV IMPACT ANALYSIS")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    
    model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping = load_model(
        checkpoint_path, device
    )
    
    inv_entity = invert_vocab(entity_vocab)
    
    # Get all civilizations (excluding special tokens)
    all_civs = [civ for civ in civ_vocab.keys() if civ not in ('<PAD>', '<UNK>')]
    all_civs = sorted(all_civs)
    
    print(f"\nFound {len(all_civs)} civilizations: {', '.join(all_civs)}")
    
    # Use a fixed map for consistency
    test_map = 'Dry Arabia'
    map_id = map_vocab.get(test_map, 0)
    print(f"Using map: {test_map} (id={map_id})")
    
    # Parameters
    num_steps = 20  # Generate 20 steps per build order
    compare_steps = 10  # Compare first 10 steps for signature
    
    print(f"\nGenerating {num_steps} steps per combination, comparing first {compare_steps}")
    print("\n" + "=" * 100)
    
    # Store results: player_civ -> {enemy_civ -> build_order_signature}
    results = defaultdict(dict)
    
    # Total combinations
    total_combinations = len(all_civs) * len(all_civs)
    current = 0
    
    print(f"\nTesting {total_combinations} combinations...\n")
    
    for player_civ_name in all_civs:
        player_civ_id = civ_vocab[player_civ_name]
        
        for enemy_civ_name in all_civs:
            current += 1
            enemy_civ_id = civ_vocab[enemy_civ_name]
            
            # Generate build order
            entities = generate_build_order(
                model=model,
                player_civ_id=player_civ_id,
                enemy_civ_id=enemy_civ_id,
                map_id=map_id,
                entity_vocab=entity_vocab,
                num_steps=num_steps,
                device=device,
                temperature=0.3,
                civ_entity_mapping=civ_entity_mapping,
                player_civ_name=player_civ_name,
                greedy=True,
                seed=42,
                exclude_special=True,
            )
            
            signature = get_build_order_signature(entities, inv_entity, compare_steps)
            results[player_civ_name][enemy_civ_name] = signature
            
            # Progress indicator
            if current % 10 == 0 or current == total_combinations:
                print(f"  Progress: {current}/{total_combinations} ({100*current/total_combinations:.1f}%)")
    
    # Analyze results
    print("\n" + "=" * 100)
    print("ANALYSIS: DOES ENEMY CIV AFFECT BUILD ORDER?")
    print("=" * 100)
    
    # For each player civ, check if different enemy civs produce different outputs
    player_impact_summary = {}
    
    for player_civ in all_civs:
        enemy_results = results[player_civ]
        unique_signatures = set(enemy_results.values())
        
        num_unique = len(unique_signatures)
        num_enemies = len(enemy_results)
        
        if num_unique == 1:
            impact = "NO IMPACT"
            symbol = "❌"
        elif num_unique == num_enemies:
            impact = "FULL IMPACT"
            symbol = "✓✓"
        else:
            impact = f"PARTIAL ({num_unique}/{num_enemies} unique)"
            symbol = "✓"
        
        player_impact_summary[player_civ] = {
            'unique': num_unique,
            'total': num_enemies,
            'impact': impact,
            'symbol': symbol
        }
    
    # Print summary table
    print(f"\n{'Player Civ':<25} {'Unique Outputs':<15} {'Enemy Civs':<15} {'Impact':<25}")
    print("-" * 80)
    
    for player_civ in all_civs:
        info = player_impact_summary[player_civ]
        print(f"{player_civ:<25} {info['unique']:<15} {info['total']:<15} {info['symbol']} {info['impact']}")
    
    # Overall summary
    civs_with_no_impact = sum(1 for p in all_civs if player_impact_summary[p]['unique'] == 1)
    civs_with_impact = len(all_civs) - civs_with_no_impact
    
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"  Civilizations where enemy civ HAS impact:    {civs_with_impact}/{len(all_civs)} ({100*civs_with_impact/len(all_civs):.1f}%)")
    print(f"  Civilizations where enemy civ has NO impact: {civs_with_no_impact}/{len(all_civs)} ({100*civs_with_no_impact/len(all_civs):.1f}%)")
    
    # Detailed breakdown for each player civ
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN BY PLAYER CIVILIZATION")
    print("=" * 100)
    
    for player_civ in all_civs:
        enemy_results = results[player_civ]
        unique_signatures = {}
        
        for enemy_civ, sig in enemy_results.items():
            if sig not in unique_signatures:
                unique_signatures[sig] = []
            unique_signatures[sig].append(enemy_civ)
        
        info = player_impact_summary[player_civ]
        print(f"\n{'─'*80}")
        print(f"PLAYER: {player_civ.upper()}")
        print(f"{'─'*80}")
        
        if len(unique_signatures) == 1:
            print(f"  ❌ Same build order against ALL {len(enemy_results)} enemy civs")
            sig = list(unique_signatures.keys())[0]
            print(f"  Build: {' → '.join(sig[:8])}...")
        else:
            print(f"  ✓ {len(unique_signatures)} different build orders depending on enemy civ:\n")
            
            for i, (sig, enemies) in enumerate(sorted(unique_signatures.items(), key=lambda x: -len(x[1])), 1):
                enemy_list = ', '.join(sorted(enemies))
                print(f"  Build Pattern {i} (vs {len(enemies)} civs: {enemy_list}):")
                print(f"    {' → '.join(sig[:8])}...")
                print()
    
    # Create a difference matrix
    print("\n" + "=" * 100)
    print("DIFFERENCE MATRIX (shows if build differs from diagonal)")
    print("=" * 100)
    print("\nLegend: ● = Same build order, ○ = Different build order")
    print(f"        (comparing {player_civ} vs {player_civ} with {player_civ} vs other)")
    print()
    
    # Header
    header = "Player \\ Enemy  │ " + " ".join(f"{c[:3]:>3}" for c in all_civs)
    print(header)
    print("─" * len(header))
    
    for player_civ in all_civs:
        # Get the "baseline" - build against same civ
        baseline_sig = results[player_civ].get(player_civ)
        
        row = f"{player_civ:15} │ "
        for enemy_civ in all_civs:
            sig = results[player_civ][enemy_civ]
            if sig == baseline_sig:
                row += " ● "
            else:
                row += " ○ "
        print(row)
    
    # Final verdict
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    
    if civs_with_no_impact == len(all_civs):
        print("\n❌ CRITICAL: Enemy civilization is completely IGNORED by the model!")
        print("   The model produces the same build order regardless of opponent.")
        print("   This suggests the enemy civ embedding is not being used effectively.")
    elif civs_with_impact == len(all_civs):
        print("\n✓✓ EXCELLENT: Enemy civilization affects ALL player civ build orders!")
        print("   The model is properly adapting to different opponents.")
    else:
        print(f"\n⚠️ MIXED: Enemy civ affects {civs_with_impact}/{len(all_civs)} player civs.")
        print("   Some civilizations adapt to opponents, others don't.")
        
        # List which ones don't adapt
        non_adapting = [p for p in all_civs if player_impact_summary[p]['unique'] == 1]
        if non_adapting:
            print(f"\n   Civs that DON'T adapt to enemy: {', '.join(non_adapting)}")
        
        adapting = [p for p in all_civs if player_impact_summary[p]['unique'] > 1]
        if adapting:
            print(f"   Civs that DO adapt to enemy: {', '.join(adapting)}")


if __name__ == '__main__':
    test_all_civ_combinations()
