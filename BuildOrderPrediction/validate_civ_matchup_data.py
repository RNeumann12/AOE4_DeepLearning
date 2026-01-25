#!/usr/bin/env python3
"""
Validation script to check if model predictions are reliable based on training data availability.

This script:
1. Analyzes the training data to count games/samples for each player_civ vs enemy_civ matchup
2. Cross-references with model predictions to identify unreliable results
3. Highlights matchups where the model might be "guessing" due to insufficient data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# Ensure repo root is on path for local imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from BuildOrderPrediction.MoE_WithDecoder_infer import (
    load_model, 
    generate_build_order, 
    invert_vocab,
)


def load_training_data(csv_path: str, wins_only: bool = True, max_seq_len: int = 50):
    """Load and filter training data similar to training script."""
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  Total rows: {len(df):,}")
    
    # Filter to wins only if specified
    if wins_only:
        df = df[df['player_won'] == 1]
        print(f"  After wins_only filter: {len(df):,}")
    
    return df


def analyze_matchup_data(df: pd.DataFrame):
    """Analyze training data to count samples per matchup."""
    
    # Count unique games per matchup
    matchup_games = df.groupby(['player_civ', 'enemy_civ'])['game_id'].nunique().reset_index()
    matchup_games.columns = ['player_civ', 'enemy_civ', 'num_games']
    
    # Count total samples (rows) per matchup
    matchup_samples = df.groupby(['player_civ', 'enemy_civ']).size().reset_index(name='num_samples')
    
    # Merge
    matchup_stats = matchup_games.merge(matchup_samples, on=['player_civ', 'enemy_civ'])
    
    # Create a pivot table for games
    games_pivot = matchup_games.pivot(index='player_civ', columns='enemy_civ', values='num_games').fillna(0).astype(int)
    
    return matchup_stats, games_pivot


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


def validate_predictions_with_data(
    model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping,
    matchup_stats, device, test_map=''
):
    """Generate predictions and cross-reference with data availability."""
    
    inv_entity = invert_vocab(entity_vocab)
    all_civs = sorted([civ for civ in civ_vocab.keys() if civ not in ('<PAD>', '<UNK>')])
    
    map_id = map_vocab.get(test_map, 0)
    
    print(f"\nGenerating predictions for all {len(all_civs)}x{len(all_civs)} matchups...")
    
    results = []
    
    for player_civ_name in all_civs:
        player_civ_id = civ_vocab[player_civ_name]
        
        for enemy_civ_name in all_civs:
            enemy_civ_id = civ_vocab[enemy_civ_name]
            
            # Generate build order
            entities = generate_build_order(
                model=model,
                player_civ_id=player_civ_id,
                enemy_civ_id=enemy_civ_id,
                map_id=map_id,
                entity_vocab=entity_vocab,
                num_steps=20,
                device=device,
                temperature=0.3,
                civ_entity_mapping=civ_entity_mapping,
                player_civ_name=player_civ_name,
                greedy=True,
                seed=42,
                exclude_special=True,
            )
            
            signature = get_build_order_signature(entities, inv_entity, 10)
            
            # Look up data availability
            matchup_data = matchup_stats[
                (matchup_stats['player_civ'] == player_civ_name) & 
                (matchup_stats['enemy_civ'] == enemy_civ_name)
            ]
            
            if len(matchup_data) > 0:
                num_games = matchup_data['num_games'].values[0]
                num_samples = matchup_data['num_samples'].values[0]
            else:
                num_games = 0
                num_samples = 0
            
            results.append({
                'player_civ': player_civ_name,
                'enemy_civ': enemy_civ_name,
                'num_games': num_games,
                'num_samples': num_samples,
                'build_signature': signature,
                'build_str': ' → '.join(signature[:8])
            })
    
    return pd.DataFrame(results)


def categorize_data_sufficiency(num_games: int) -> str:
    """Categorize data sufficiency level."""
    if num_games == 0:
        return "NO_DATA"
    elif num_games < 5:
        return "VERY_LOW"
    elif num_games < 20:
        return "LOW"
    elif num_games < 50:
        return "MODERATE"
    elif num_games < 100:
        return "GOOD"
    else:
        return "EXCELLENT"


def main():
    parser = argparse.ArgumentParser(description='Validate model predictions against training data availability')
    parser.add_argument('--csv_path', type=str, default='training_data_2026_01.csv',
                       help='Path to training data CSV')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--wins_only', action='store_true', default=True,
                       help='Filter to wins only (as in training)')
    parser.add_argument('--max_seq_len', type=int, default=50,
                       help='Max sequence length (as in training)')
    parser.add_argument('--map', type=str, default='',
                       help='Map to use for generation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, cuda')
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 100)
    print("VALIDATION: Cross-referencing Model Predictions with Training Data")
    print("=" * 100)
    
    # Load training data
    df = load_training_data(args.csv_path, wins_only=args.wins_only, max_seq_len=args.max_seq_len)
    
    # Analyze matchups
    matchup_stats, games_pivot = analyze_matchup_data(df)
    
    # Load model
    print("\nLoading model...")
    model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping = load_model(
        args.checkpoint, device
    )
    
    # Get all civs from vocab
    all_civs = sorted([civ for civ in civ_vocab.keys() if civ not in ('<PAD>', '<UNK>')])
    
    # Print data availability summary
    print("\n" + "=" * 100)
    print("TRAINING DATA AVAILABILITY BY MATCHUP")
    print("=" * 100)
    
    print(f"\nTotal unique matchups in data: {len(matchup_stats)}")
    print(f"Total possible matchups: {len(all_civs) * len(all_civs)}")
    print(f"Missing matchups: {len(all_civs) * len(all_civs) - len(matchup_stats)}")
    
    # Games per player civ
    print("\n--- Games per Player Civilization ---")
    player_games = df.groupby('player_civ')['game_id'].nunique().sort_values(ascending=False)
    print(f"{'Civilization':<25} {'Games':>10}")
    print("-" * 37)
    for civ, games in player_games.items():
        print(f"{civ:<25} {games:>10,}")
    
    # Games per matchup distribution
    print("\n--- Matchup Data Sufficiency Distribution ---")
    matchup_stats['sufficiency'] = matchup_stats['num_games'].apply(categorize_data_sufficiency)
    sufficiency_counts = matchup_stats['sufficiency'].value_counts()
    
    categories = ['NO_DATA', 'VERY_LOW', 'LOW', 'MODERATE', 'GOOD', 'EXCELLENT']
    print(f"{'Category':<15} {'Games':<15} {'Count':>10} {'%':>10}")
    print("-" * 50)
    for cat in categories:
        count = sufficiency_counts.get(cat, 0)
        pct = 100 * count / len(matchup_stats) if len(matchup_stats) > 0 else 0
        games_range = {
            'NO_DATA': '0',
            'VERY_LOW': '1-4',
            'LOW': '5-19',
            'MODERATE': '20-49',
            'GOOD': '50-99',
            'EXCELLENT': '100+'
        }[cat]
        print(f"{cat:<15} {games_range:<15} {count:>10} {pct:>9.1f}%")
    
    # Add missing matchups with 0 games
    all_matchups = []
    for p_civ in all_civs:
        for e_civ in all_civs:
            existing = matchup_stats[
                (matchup_stats['player_civ'] == p_civ) & 
                (matchup_stats['enemy_civ'] == e_civ)
            ]
            if len(existing) == 0:
                all_matchups.append({
                    'player_civ': p_civ,
                    'enemy_civ': e_civ,
                    'num_games': 0,
                    'num_samples': 0,
                    'sufficiency': 'NO_DATA'
                })
    
    if all_matchups:
        missing_df = pd.DataFrame(all_matchups)
        matchup_stats = pd.concat([matchup_stats, missing_df], ignore_index=True)
    
    # Validate predictions
    print("\n" + "=" * 100)
    print("GENERATING AND VALIDATING PREDICTIONS")
    print("=" * 100)
    
    results_df = validate_predictions_with_data(
        model, entity_vocab, civ_vocab, map_vocab, civ_entity_mapping,
        matchup_stats, device, test_map=args.map
    )
    
    # Add sufficiency category
    results_df['sufficiency'] = results_df['num_games'].apply(categorize_data_sufficiency)
    
    # Analyze: Do predictions differ by data availability?
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS: BUILD ORDER CONSISTENCY BY DATA SUFFICIENCY")
    print("=" * 100)
    
    # For each player civ, analyze if predictions vary with data availability
    for player_civ in all_civs:
        player_results = results_df[results_df['player_civ'] == player_civ]
        
        # Group by build signature
        unique_builds = player_results.groupby('build_str')['enemy_civ'].apply(list).to_dict()
        
        print(f"\n{'─'*80}")
        print(f"PLAYER: {player_civ.upper()}")
        print(f"{'─'*80}")
        
        if len(unique_builds) == 1:
            # Same build for all enemies
            build = list(unique_builds.keys())[0]
            data_summary = player_results.groupby('sufficiency')['enemy_civ'].count().to_dict()
            total_games = player_results['num_games'].sum()
            avg_games = player_results['num_games'].mean()
            
            print(f"  ⚠️  Same build against ALL {len(player_results)} enemy civs")
            print(f"  Build: {build}...")
            print(f"  Data: {total_games:,} total games, avg {avg_games:.1f} games/matchup")
            
            # Check if this is due to lack of diverse data
            no_data = data_summary.get('NO_DATA', 0)
            very_low = data_summary.get('VERY_LOW', 0)
            low_data = no_data + very_low
            
            if low_data > len(player_results) * 0.5:
                print(f"  ⚠️  WARNING: {low_data}/{len(player_results)} matchups have LOW/NO data!")
                print(f"      Model may be generalizing from limited data.")
            else:
                print(f"  ✓  Data coverage looks reasonable - this may be an optimal universal build.")
        else:
            print(f"  ✓  {len(unique_builds)} different builds depending on enemy civ:\n")
            
            for build_str, enemies in sorted(unique_builds.items(), key=lambda x: -len(x[1])):
                # Get data stats for these enemies
                enemy_data = player_results[player_results['enemy_civ'].isin(enemies)]
                total_games = enemy_data['num_games'].sum()
                avg_games = enemy_data['num_games'].mean()
                no_data_enemies = enemy_data[enemy_data['num_games'] == 0]['enemy_civ'].tolist()
                low_data_enemies = enemy_data[enemy_data['num_games'] < 5]['enemy_civ'].tolist()
                
                print(f"  Build Pattern (vs {len(enemies)} civs):")
                print(f"    {build_str}...")
                print(f"    Data: {total_games:,} total games, avg {avg_games:.1f} games/matchup")
                
                if no_data_enemies:
                    print(f"    ⚠️  NO DATA for: {', '.join(no_data_enemies[:5])}" + 
                          (f" (+{len(no_data_enemies)-5} more)" if len(no_data_enemies) > 5 else ""))
                
                # Check if this build is driven by specific high-data matchups
                high_data = enemy_data[enemy_data['num_games'] >= 20]
                if len(high_data) > 0:
                    print(f"    ✓  High-data matchups: {', '.join(high_data['enemy_civ'].tolist()[:5])}")
                print()
    
    # Summary: Reliability assessment
    print("\n" + "=" * 100)
    print("RELIABILITY ASSESSMENT SUMMARY")
    print("=" * 100)
    
    # Calculate reliability metrics
    reliable_matchups = results_df[results_df['num_games'] >= 20]
    questionable_matchups = results_df[(results_df['num_games'] > 0) & (results_df['num_games'] < 20)]
    no_data_matchups = results_df[results_df['num_games'] == 0]
    
    print(f"\n{'Category':<30} {'Matchups':>12} {'%':>10}")
    print("-" * 55)
    print(f"{'HIGH RELIABILITY (≥20 games)':<30} {len(reliable_matchups):>12} {100*len(reliable_matchups)/len(results_df):>9.1f}%")
    print(f"{'QUESTIONABLE (1-19 games)':<30} {len(questionable_matchups):>12} {100*len(questionable_matchups)/len(results_df):>9.1f}%")
    print(f"{'NO DATA (0 games)':<30} {len(no_data_matchups):>12} {100*len(no_data_matchups)/len(results_df):>9.1f}%")
    
    # Identify potentially problematic predictions
    print("\n" + "=" * 100)
    print("POTENTIALLY UNRELIABLE PREDICTIONS (0 games in training)")
    print("=" * 100)
    
    if len(no_data_matchups) > 0:
        print(f"\nThese {len(no_data_matchups)} matchups have NO training data - predictions are pure generalization:")
        print(f"\n{'Player Civ':<25} {'Enemy Civ':<25} {'Predicted Build (first 4 steps)':<50}")
        print("-" * 100)
        
        for _, row in no_data_matchups.head(30).iterrows():
            build_preview = ' → '.join(row['build_signature'][:4]) if row['build_signature'] else "(none)"
            print(f"{row['player_civ']:<25} {row['enemy_civ']:<25} {build_preview:<50}")
        
        if len(no_data_matchups) > 30:
            print(f"\n... and {len(no_data_matchups) - 30} more matchups with no data")
    else:
        print("\n✓ All matchups have some training data!")
    
    # Check for "suspicious" patterns: same build across very different data levels
    print("\n" + "=" * 100)
    print("SUSPICIOUS PATTERNS: Same build despite varied data availability")
    print("=" * 100)
    
    suspicious_civs = []
    for player_civ in all_civs:
        player_results = results_df[results_df['player_civ'] == player_civ]
        unique_builds = player_results['build_str'].nunique()
        
        if unique_builds == 1:
            # Same build for all - check data variance
            has_high_data = (player_results['num_games'] >= 50).any()
            has_no_data = (player_results['num_games'] == 0).any()
            
            if has_high_data and has_no_data:
                suspicious_civs.append(player_civ)
    
    if suspicious_civs:
        print(f"\n⚠️  These civs produce identical builds for matchups with BOTH high AND no data:")
        for civ in suspicious_civs:
            civ_data = results_df[results_df['player_civ'] == civ]
            high = civ_data[civ_data['num_games'] >= 50]['enemy_civ'].tolist()
            none = civ_data[civ_data['num_games'] == 0]['enemy_civ'].tolist()
            print(f"\n  {civ}:")
            print(f"    High data vs: {', '.join(high[:5])}" + (f" (+{len(high)-5} more)" if len(high) > 5 else ""))
            print(f"    No data vs: {', '.join(none[:5])}" + (f" (+{len(none)-5} more)" if len(none) > 5 else ""))
            print(f"    → Model is likely applying a generic build pattern")
    else:
        print("\n✓ No suspicious patterns detected!")
    
    # Final verdict
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    
    reliability_score = len(reliable_matchups) / len(results_df) * 100
    
    if reliability_score >= 80:
        print(f"\n✓✓ HIGH RELIABILITY: {reliability_score:.1f}% of matchups have sufficient training data")
    elif reliability_score >= 50:
        print(f"\n✓  MODERATE RELIABILITY: {reliability_score:.1f}% of matchups have sufficient training data")
    elif reliability_score >= 20:
        print(f"\n⚠️  LOW RELIABILITY: Only {reliability_score:.1f}% of matchups have sufficient training data")
    else:
        print(f"\n❌ VERY LOW RELIABILITY: Only {reliability_score:.1f}% of matchups have sufficient training data")
    
    print(f"\nRecommendations:")
    if len(no_data_matchups) > 0:
        print(f"  - {len(no_data_matchups)} matchups have no data - consider collecting more games")
    if len(questionable_matchups) > 0:
        print(f"  - {len(questionable_matchups)} matchups have <20 games - predictions may be noisy")
    if suspicious_civs:
        print(f"  - {len(suspicious_civs)} civs show identical builds despite data variance - may need more diverse training")
    
    # Save detailed results
    output_path = 'matchup_validation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
