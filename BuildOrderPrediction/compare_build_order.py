#!/usr/bin/env python3
"""Compare a generated build order against training data and report win rates.

This script finds games in the training data that have similar build orders
to a generated/input build order, and reports win rate statistics.

Example:
  python BuildOrderPrediction/compare_build_order.py \
    --build_order BuildOrderPrediction/Predictions/english_vs_french_dry_arabia.txt \
    --training_csv training_data_2026_01.csv \
    --player_civ English \
    --enemy_civ French \
    --map "Dry Arabia" \
    --tolerance 3

  # Or provide build order as comma-separated string:
  python BuildOrderPrediction/compare_build_order.py \
    --build_order_str "Villager,House,Villager,Gold Mining Camp,Villager" \
    --training_csv training_data_2026_01.csv \
    --player_civ English
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def parse_build_order_file(filepath: str) -> List[str]:
    """Parse a build order from a text file.
    
    Expected format (one entity per line, with optional step numbers):
        1      Villager
        2      House
        3      Villager
        ...
    
    Returns:
        List of entity names in order
    """
    entities = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip header/separator lines
            if line.startswith('=') or line.startswith('-') or line.startswith('Step'):
                continue
            
            # Parse "step_num    entity_name" format
            # Handle both tab and multiple space separators
            parts = re.split(r'\s{2,}|\t', line)
            if len(parts) >= 2:
                # First part is step number, rest is entity name
                entity = parts[1].strip()
                if entity and not entity.startswith('<'):  # Skip special tokens
                    entities.append(entity)
            elif len(parts) == 1:
                # Just entity name
                entity = parts[0].strip()
                # Skip if it's just a number (step count line)
                if entity and not entity.isdigit() and not entity.startswith('<'):
                    entities.append(entity)
    
    return entities


def parse_build_order_string(build_str: str) -> List[str]:
    """Parse a comma-separated build order string."""
    return [e.strip() for e in build_str.split(',') if e.strip()]


def load_training_data(
    csv_path: str,
    player_civ: Optional[str] = None,
    enemy_civ: Optional[str] = None,
    map_name: Optional[str] = None,
    event_filter: List[str] = None,
    skip_starting_entities: bool = True,
    min_time: int = 1
) -> pd.DataFrame:
    """Load training data and optionally filter by conditions.
    
    Args:
        csv_path: Path to training CSV file
        player_civ: Filter by player civilization (case-insensitive)
        enemy_civ: Filter by enemy civilization (case-insensitive)
        map_name: Filter by map name (case-insensitive)
        event_filter: List of event types to include (default: ['FINISH', 'BUILD'])
        skip_starting_entities: If True, skip entities at time=0 (starting units)
        min_time: Minimum time threshold to filter starting entities
    
    Returns:
        Filtered DataFrame
    """
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df):,}")
    
    # Apply event filter
    if event_filter is None:
        event_filter = ['FINISH', 'BUILD']
    df = df[df['event'].isin(event_filter)]
    print(f"  After event filter ({event_filter}): {len(df):,}")
    
    # Skip starting entities (time=0) - these are initial units, not build orders
    if skip_starting_entities:
        df = df[df['time'] >= min_time]
        print(f"  After skipping starting entities (time >= {min_time}): {len(df):,}")
    
    # Apply civilization filters (case-insensitive)
    if player_civ:
        player_civ_lower = player_civ.lower()
        df = df[df['player_civ'].str.lower() == player_civ_lower]
        print(f"  After player_civ filter ({player_civ}): {len(df):,}")
    
    if enemy_civ:
        enemy_civ_lower = enemy_civ.lower()
        df = df[df['enemy_civ'].str.lower() == enemy_civ_lower]
        print(f"  After enemy_civ filter ({enemy_civ}): {len(df):,}")
    
    if map_name:
        map_name_lower = map_name.lower()
        df = df[df['map'].str.lower() == map_name_lower]
        print(f"  After map filter ({map_name}): {len(df):,}")
    
    return df


def extract_game_build_orders(df: pd.DataFrame, max_steps: int = None) -> Dict[Tuple, dict]:
    """Extract build orders for each game in the dataset.
    
    Args:
        df: Training DataFrame with game_id, profile_id, entity, player_won columns
        max_steps: Max number of steps to consider (None = all)
    
    Returns:
        Dict mapping (game_id, profile_id) -> {
            'entities': List[str],
            'won': bool,
            'player_civ': str,
            'enemy_civ': str,
            'map': str
        }
    """
    games = {}
    
    # Group by game and player
    grouped = df.groupby(['game_id', 'profile_id'])
    
    for (game_id, profile_id), group in grouped:
        # Sort by time
        group = group.sort_values('time')
        
        # Extract entities
        entities = group['entity'].tolist()
        if max_steps:
            entities = entities[:max_steps]
        
        # Get game outcome
        won = group['player_won'].iloc[0] == 1
        
        games[(game_id, profile_id)] = {
            'entities': entities,
            'won': won,
            'player_civ': group['player_civ'].iloc[0],
            'enemy_civ': group['enemy_civ'].iloc[0],
            'map': group['map'].iloc[0]
        }
    
    return games


def compute_similarity(
    target: List[str],
    candidate: List[str],
    method: str = 'prefix'
) -> Tuple[float, int, int]:
    """Compute similarity between two build orders.
    
    Args:
        target: Target build order (generated)
        candidate: Candidate build order (from training data)
        method: Similarity method ('prefix', 'lcs', 'edit')
    
    Returns:
        (similarity_score, matching_steps, total_compared)
    """
    if method == 'prefix':
        # Count matching prefix (first N entities that match exactly)
        matches = 0
        compare_len = min(len(target), len(candidate))
        for i in range(compare_len):
            if target[i] == candidate[i]:
                matches += 1
            else:
                break
        return matches / len(target) if target else 0.0, matches, len(target)
    
    elif method == 'lcs':
        # Longest Common Subsequence
        m, n = len(target), len(candidate)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if target[i-1] == candidate[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_len = dp[m][n]
        return lcs_len / len(target) if target else 0.0, lcs_len, len(target)
    
    elif method == 'edit':
        # Levenshtein distance (normalized to similarity)
        m, n = len(target), len(candidate)
        if m == 0:
            return 0.0, 0, 0
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if target[i-1] == candidate[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_dist = dp[m][n]
        similarity = 1.0 - (edit_dist / max(m, n))
        return similarity, m - edit_dist, m
    
    elif method == 'positional':
        # Count entities at matching positions
        compare_len = min(len(target), len(candidate))
        matches = sum(1 for i in range(compare_len) if target[i] == candidate[i])
        return matches / len(target) if target else 0.0, matches, len(target)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def find_similar_builds(
    target_build: List[str],
    games: Dict[Tuple, dict],
    method: str = 'lcs',
    min_similarity: float = 0.5,
    tolerance: int = None
) -> List[dict]:
    """Find games with similar build orders.
    
    Args:
        target_build: Target build order to compare against
        games: Dict of games from extract_game_build_orders
        method: Similarity method
        min_similarity: Minimum similarity threshold (0-1)
        tolerance: Alternative to min_similarity - max number of differences allowed
    
    Returns:
        List of matching games with similarity scores
    """
    matches = []
    
    for (game_id, profile_id), game_info in games.items():
        candidate = game_info['entities'][:len(target_build)]
        
        similarity, match_count, total = compute_similarity(
            target_build, candidate, method
        )
        
        # Check threshold
        if tolerance is not None:
            # Tolerance-based: allow up to N differences
            differences = len(target_build) - match_count
            if differences <= tolerance:
                matches.append({
                    'game_id': game_id,
                    'profile_id': profile_id,
                    'similarity': similarity,
                    'match_count': match_count,
                    'differences': differences,
                    'won': game_info['won'],
                    'player_civ': game_info['player_civ'],
                    'enemy_civ': game_info['enemy_civ'],
                    'map': game_info['map'],
                    'build_order': candidate
                })
        elif similarity >= min_similarity:
            matches.append({
                'game_id': game_id,
                'profile_id': profile_id,
                'similarity': similarity,
                'match_count': match_count,
                'differences': len(target_build) - match_count,
                'won': game_info['won'],
                'player_civ': game_info['player_civ'],
                'enemy_civ': game_info['enemy_civ'],
                'map': game_info['map'],
                'build_order': candidate
            })
    
    # Sort by similarity descending
    matches.sort(key=lambda x: -x['similarity'])
    
    return matches


def compute_win_rate_stats(matches: List[dict]) -> dict:
    """Compute win rate statistics for matched games."""
    if not matches:
        return {
            'total_matches': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_similarity': 0.0
        }
    
    wins = sum(1 for m in matches if m['won'])
    losses = len(matches) - wins
    
    return {
        'total_matches': len(matches),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(matches) * 100,
        'avg_similarity': sum(m['similarity'] for m in matches) / len(matches),
        'min_similarity': min(m['similarity'] for m in matches),
        'max_similarity': max(m['similarity'] for m in matches)
    }


def print_results(
    target_build: List[str],
    matches: List[dict],
    stats: dict,
    show_top: int = 10,
    show_builds: bool = False
):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("BUILD ORDER COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\nTarget Build Order ({len(target_build)} steps):")
    print("-" * 50)
    for i, entity in enumerate(target_build[:20], 1):
        print(f"  {i:>3}. {entity}")
    if len(target_build) > 20:
        print(f"  ... and {len(target_build) - 20} more steps")
    
    print("\n" + "=" * 70)
    print("WIN RATE STATISTICS")
    print("=" * 70)
    
    if stats['total_matches'] == 0:
        print("\n  ⚠ No matching games found!")
        print("  Try:")
        print("    - Lowering --min_similarity threshold")
        print("    - Increasing --tolerance value")
        print("    - Removing --map filter")
        return
    
    print(f"\n  Matching Games:     {stats['total_matches']}")
    print(f"  Wins:               {stats['wins']}")
    print(f"  Losses:             {stats['losses']}")
    print(f"\n  ★ WIN RATE:         {stats['win_rate']:.1f}%")
    print(f"\n  Similarity Range:   {stats['min_similarity']:.1%} - {stats['max_similarity']:.1%}")
    print(f"  Average Similarity: {stats['avg_similarity']:.1%}")
    
    # Interpretation
    print("\n  Interpretation:")
    if stats['win_rate'] >= 60:
        print("    ✓ Strong build order - Above average win rate")
    elif stats['win_rate'] >= 50:
        print("    ~ Solid build order - Balanced win rate")
    elif stats['win_rate'] >= 40:
        print("    ! Weak build order - Below average win rate")
    else:
        print("    ✗ Poor build order - Low win rate, consider alternatives")
    
    # Show top matches
    if matches and show_top > 0:
        print(f"\n" + "=" * 70)
        print(f"TOP {min(show_top, len(matches))} SIMILAR GAMES")
        print("=" * 70)
        print(f"\n{'Similarity':<12} {'Result':<8} {'Matchup':<30} {'Map':<20}")
        print("-" * 70)
        
        for match in matches[:show_top]:
            result = "WIN" if match['won'] else "LOSS"
            matchup = f"{match['player_civ']} vs {match['enemy_civ']}"
            print(f"{match['similarity']:>9.1%}   {result:<8} {matchup:<30} {match['map']:<20}")
            
            if show_builds:
                print(f"           Build: {', '.join(match['build_order'][:10])}...")
    
    # Win rate by similarity bracket
    if len(matches) >= 10:
        print(f"\n" + "=" * 70)
        print("WIN RATE BY SIMILARITY BRACKET")
        print("=" * 70)
        
        brackets = [
            (0.9, 1.0, "90-100%"),
            (0.8, 0.9, "80-90%"),
            (0.7, 0.8, "70-80%"),
            (0.6, 0.7, "60-70%"),
            (0.5, 0.6, "50-60%"),
            (0.0, 0.5, "<50%")
        ]
        
        print(f"\n{'Bracket':<12} {'Games':<10} {'Win Rate':<12}")
        print("-" * 40)
        
        for low, high, label in brackets:
            bracket_matches = [m for m in matches if low <= m['similarity'] < high]
            if bracket_matches:
                bracket_wins = sum(1 for m in bracket_matches if m['won'])
                bracket_wr = bracket_wins / len(bracket_matches) * 100
                print(f"{label:<12} {len(bracket_matches):<10} {bracket_wr:.1f}%")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Compare a build order against training data and report win rates'
    )
    
    # Build order input (one of these required)
    parser.add_argument('--build_order', type=str,
                       help='Path to build order text file')
    parser.add_argument('--build_order_str', type=str,
                       help='Comma-separated build order string')
    
    # Training data
    parser.add_argument('--training_csv', type=str, required=True,
                       help='Path to training CSV file')
    
    # Filters
    parser.add_argument('--player_civ', type=str,
                       help='Filter by player civilization')
    parser.add_argument('--enemy_civ', type=str,
                       help='Filter by enemy civilization')
    parser.add_argument('--map', type=str,
                       help='Filter by map name')
    
    # Similarity settings
    parser.add_argument('--method', type=str, default='lcs',
                       choices=['prefix', 'lcs', 'edit', 'positional'],
                       help='Similarity method (default: lcs)')
    parser.add_argument('--min_similarity', type=float, default=0.6,
                       help='Minimum similarity threshold 0-1 (default: 0.6)')
    parser.add_argument('--tolerance', type=int, default=None,
                       help='Max allowed differences (alternative to min_similarity)')
    
    # Output settings
    parser.add_argument('--show_top', type=int, default=10,
                       help='Number of top matches to show (default: 10)')
    parser.add_argument('--show_builds', action='store_true',
                       help='Show build orders for top matches')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Max build steps to compare (default: use all from input)')
    
    # Export
    parser.add_argument('--export_csv', type=str,
                       help='Export matches to CSV file')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.build_order and not args.build_order_str:
        print("Error: Must provide either --build_order or --build_order_str")
        sys.exit(1)
    
    # Parse target build order
    if args.build_order:
        target_build = parse_build_order_file(args.build_order)
        print(f"Loaded build order from: {args.build_order}")
    else:
        target_build = parse_build_order_string(args.build_order_str)
        print(f"Parsed build order from string")
    
    if args.max_steps:
        target_build = target_build[:args.max_steps]
    
    print(f"  Build length: {len(target_build)} steps")
    
    if not target_build:
        print("Error: No valid entities found in build order")
        sys.exit(1)
    
    # Load training data
    df = load_training_data(
        args.training_csv,
        player_civ=args.player_civ,
        enemy_civ=args.enemy_civ,
        map_name=args.map
    )
    
    if df.empty:
        print("Error: No training data after filtering")
        sys.exit(1)
    
    # Extract game build orders
    print("\nExtracting game build orders...")
    games = extract_game_build_orders(df, max_steps=len(target_build) + 10)
    print(f"  Found {len(games)} unique games")
    
    # Find similar builds
    print(f"\nFinding similar builds (method={args.method})...")
    matches = find_similar_builds(
        target_build,
        games,
        method=args.method,
        min_similarity=args.min_similarity,
        tolerance=args.tolerance
    )
    
    # Compute statistics
    stats = compute_win_rate_stats(matches)
    
    # Print results
    print_results(
        target_build,
        matches,
        stats,
        show_top=args.show_top,
        show_builds=args.show_builds
    )
    
    # Export if requested
    if args.export_csv and matches:
        export_df = pd.DataFrame([
            {
                'game_id': m['game_id'],
                'profile_id': m['profile_id'],
                'similarity': m['similarity'],
                'match_count': m['match_count'],
                'differences': m['differences'],
                'won': m['won'],
                'player_civ': m['player_civ'],
                'enemy_civ': m['enemy_civ'],
                'map': m['map']
            }
            for m in matches
        ])
        export_df.to_csv(args.export_csv, index=False)
        print(f"\nExported {len(matches)} matches to: {args.export_csv}")


if __name__ == '__main__':
    main()
