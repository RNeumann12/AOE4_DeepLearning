"""Post-training cluster analysis: compute event distributions by age for each cluster."""
import argparse
import os
import pandas as pd
import numpy as np

# optional wandb
try:
    import wandb
    _wandb_available = True
except Exception:
    wandb = None
    _wandb_available = False


def analyze_clusters(csv_path, cluster_csv_path, args, out_dir="outputs/unsup", use_wandb=False):
    """
    Analyze clusters by computing event distribution across ages.
    
    Args:
        csv_path: Path to original training CSV (with events, ages, etc.)
        cluster_csv_path: Path to cluster assignments (game_id, player_id, cluster)
        out_dir: Output directory
        use_wandb: Log results to W&B
    """

    # Load data
    print(f"Loading training data from {csv_path}...")
    df_data = pd.read_csv(csv_path)
    
    print(f"Loading cluster assignments from {cluster_csv_path}...")
    df_clusters = pd.read_csv(cluster_csv_path)
    
    # Merge data with cluster assignments
    print("Merging data with cluster assignments...")
    df_merged = df_data.merge(
        df_clusters[['game_id', 'player_id', 'cluster']],
        left_on=['game_id', 'profile_id'],
        right_on=['game_id', 'player_id'],
        how='left'
    )
    
    # Remove rows without cluster assignment
    df_merged = df_merged.dropna(subset=['cluster'])
    df_merged['cluster'] = df_merged['cluster'].astype(int)
    
    print(f"Merged {len(df_merged)} rows across {len(df_clusters)} games/players")
    
    # Analyze each cluster
    clusters = sorted(df_merged['cluster'].unique())
    print(f"\nAnalyzing {len(clusters)} clusters...\n")
    
    cluster_summaries = []
    
    for cluster_id in clusters:
        df_c = df_merged[df_merged['cluster'] == cluster_id]
        n_games = df_c.groupby(['game_id', 'profile_id']).ngroups
        
        print(f"Cluster {cluster_id}: {n_games} games")
        summary = {'cluster': cluster_id, 'n_games': n_games}
        
        # Average entity counts by age per game across all games in cluster
        ages = ['DARK', 'FEUDAL', 'CASTLE', 'IMPERIAL']
        games = df_c[['game_id', 'profile_id']].drop_duplicates()
        n_games = len(games)
        for age in ages:
            df_age = df_c[df_c['age'] == age]

            # count entities per game
            game_entity_counts = (
                df_age.groupby(['game_id', 'profile_id', 'entity'])
                .size()
                .unstack(fill_value=0)  # <-- critical
            )

            # ensure all games are present (even if age never reached)
            game_entity_counts = (
                games.set_index(['game_id', 'profile_id'])
                .join(game_entity_counts, how='left')
                .fillna(0)
            )

            # ---- entity averages ----
            for entity_type in game_entity_counts.columns:
                avg_per_game = game_entity_counts[entity_type].mean()
                summary[f"{age}_{entity_type}_avg"] = round(float(avg_per_game), 2)
                print(f"  {age} {entity_type}: {avg_per_game:.2f} avg per game")

            # ---- total entities per game ----
            total_entities_per_game = game_entity_counts.sum(axis=1).mean()
            summary[f"{age}_total_entities_avg"] = round(float(total_entities_per_game), 2)
            print(f"  {age} total entities: {total_entities_per_game:.2f} avg per game")

            # ---- villagers ----
            if 'Villager' in game_entity_counts.columns:
                villager_avg = game_entity_counts['Villager'].mean()
            else:
                villager_avg = 0.0

            summary[f"{age}_villager_avg"] = round(float(villager_avg), 2)
            print(f"  {age} villagers: {villager_avg:.2f} avg per game")
        
        # Map distribution
        print(f"  Map distribution:")
        if 'map' in df_c.columns:
            unique_games = (
                df_c[['game_id', 'profile_id', 'map']]
                .drop_duplicates(subset=['game_id', 'profile_id'])
            )
            # map_counts = df_c.groupby(['game_id', 'profile_id', 'map']).size().reset_index(name='count')
            map_dist = unique_games['map'].value_counts()
            # summed = 
            for map_name, count in map_dist.items():
                summary[f"map_{map_name}"] = int(count)
                # pct = round(100 * count / n_games, 1)
                print(f"    {map_name}: {count} games")
        
        # Civ distribution
        print(f"  Civ distribution:")
        if 'player_civ' in df_c.columns:
            unique_games = (
                df_c[['game_id', 'profile_id', 'player_civ']]
                .drop_duplicates(subset=['game_id', 'profile_id'])
            )
            civ_dist = unique_games['player_civ'].value_counts()
            for civ_name, count in civ_dist.items():
                summary[f"civ_{civ_name}"] = int(count)
                # pct = round(100 * count / n_games, 1)
                print(f"    {civ_name}: {count} games")
            
        cluster_summaries.append(summary)
    
    # Create summary table
    df_summary = pd.DataFrame(cluster_summaries)
    count_cols = df_summary.filter(regex="^(map_|civ_|n_games$)").columns
    df_summary[count_cols] = df_summary[count_cols].fillna(0).astype(int)

    # Reorder columns: cluster, n_games, maps, civs, then ages (DARK, FEUDAL, CASTLE, IMPERIAL)
    cols_ordered = ['cluster', 'n_games']
    
    # Add map columns
    map_cols = [c for c in df_summary.columns if c.startswith('map_')]
    cols_ordered.extend(sorted(map_cols))
    
    # Add civ columns
    civ_cols = [c for c in df_summary.columns if c.startswith('civ_')]
    cols_ordered.extend(sorted(civ_cols))
    
    # Add age columns in order: DARK, FEUDAL, CASTLE, IMPERIAL
    # Priority: summary columns first (total_entities_avg, villager_avg), then individual entity columns
    for age in ['DARK', 'FEUDAL', 'CASTLE', 'IMPERIAL']:
        age_all_cols = [c for c in df_summary.columns if c.startswith(age + '_')]
        
        # Separate summary columns and entity columns
        summary_cols = [c for c in age_all_cols if 'total_entities_avg' in c or 'villager_avg' in c]
        entity_cols = [c for c in age_all_cols if 'total_entities_avg' not in c and 'villager_avg' not in c]
        
        # Add in order: total_entities_avg, villager_avg, then entity columns
        cols_ordered.extend(sorted(summary_cols))
        cols_ordered.extend(sorted(entity_cols))
    
    # Only keep columns that exist
    cols_ordered = [c for c in cols_ordered if c in df_summary.columns]
    df_summary = df_summary[cols_ordered]
    summary_path = os.path.join(out_dir, "cluster_event_summary.csv")
    df_summary.to_csv(summary_path, sep=";",decimal=",", index=False)
    print(f"\nSummary saved to {summary_path}")
    
    # Log to W&B if available
    if use_wandb and _wandb_available:
        # Log as table
        table = wandb.Table(dataframe=df_summary)
        wandb.log({"cluster_event_summary": table})
        
        # Also log per-cluster metrics
        for _, row in df_summary.iterrows():
            cluster_id = int(row['cluster'])
            log_dict = {f"cluster_{cluster_id}_n_games": int(row['n_games'])}
            
            # Add age-entity averages
            for col in row.index:
                if col not in ['cluster', 'n_games'] and pd.notna(row[col]):
                    log_dict[f"cluster_{cluster_id}_{col}"] = row[col]
            
            wandb.log(log_dict)
        
        wandb.finish()
        print("Logged to W&B")
    
    return df_summary


def cli():
    parser = argparse.ArgumentParser(description="Analyze cluster event distributions across ages")
    parser.add_argument("--npz", type=str, default="aoe4_dataset.npz", help="Path to NPZ dataset")
    parser.add_argument("--csv", type=str, default="input_event_based.csv",
                        help="Path to training CSV with events and ages")
    parser.add_argument("--clusters", type=str, default="outputs/unsup/cluster_labels_kmeans.csv",
                        help="Path to cluster assignments CSV")
    parser.add_argument("--out_dir", type=str, default="outputs/unsup",
                        help="Output directory for results")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--wandb_project", type=str, default="aoe4-strat",
                        help="W&B project name")
    
    args = parser.parse_args()
    
    # analyze_clusters(
    #     csv_path=args.csv,
    #     cluster_csv_path=args.clusters,
    #     args=args,
    #     out_dir=args.out_dir,
    #     use_wandb=args.wandb
    # )

    build_game_level_table(
        csv_path=args.csv,
        cluster_csv_path=args.clusters,
        out_dir=args.out_dir
    )

def build_game_level_table(csv_path, cluster_csv_path, out_dir="outputs/unsup"):
    """
    Build one row per (game_id, profile_id) containing:
    cluster, civ, map, and per-age entity counts.
    """

    print("Loading data...")
    df_data = pd.read_csv(csv_path)
    df_clusters = pd.read_csv(cluster_csv_path)

    print("Merging...")
    df = df_data.merge(
        df_clusters[['game_id', 'player_id', 'cluster']],
        left_on=['game_id', 'profile_id'],
        right_on=['game_id', 'player_id'],
        how='left'
    )

    df = df.dropna(subset=['cluster'])
    df['cluster'] = df['cluster'].astype(int)

    # ---- Base game info (one row per game/player) ----
    base = (
        df[['game_id', 'profile_id', 'cluster', 'player_civ', 'map']]
        .drop_duplicates(subset=['game_id', 'profile_id'])
    )

    # ---- Count entities per age per game ----
    entity_counts = (
        df.groupby(['game_id', 'profile_id', 'age', 'entity'])
          .size()
          .reset_index(name='count')
    )

    # Pivot to wide format
    entity_pivot = (
        entity_counts
        .pivot_table(
            index=['game_id', 'profile_id'],
            columns=['age', 'entity'],
            values='count',
            fill_value=0
        )
    )

    # Flatten MultiIndex columns
    entity_pivot.columns = [
        f"{age}_{entity}"
        for age, entity in entity_pivot.columns
    ]

    entity_pivot = entity_pivot.reset_index()

    # ---- Merge ----
    final_df = base.merge(
        entity_pivot,
        on=['game_id', 'profile_id'],
        how='left'
    )

    # Replace NaN with 0 for entity columns
    count_cols = final_df.columns.difference(
        ['game_id', 'profile_id', 'cluster', 'player_civ', 'map']
    )

    final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

    # ---- Save ----
    path = os.path.join(out_dir, "cluster_game_level_table.csv")
    final_df.to_csv(path, sep=";", decimal=",", index=False)

    print(f"Game-level table saved to {path}")

    return final_df



if __name__ == "__main__":
    cli()
