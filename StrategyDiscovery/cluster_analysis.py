"""
Cluster Analysis & Strategy Discovery
======================================
This script analyzes the clusters from the GRU autoencoder to discover and validate
game strategies.

It provides:
1. Cluster profiling (average stats, key features)
2. Temporal pattern analysis (event sequences, timing)
3. Statistical validation (win rates, performance metrics)
4. Visualization of cluster characteristics
5. W&B logging for all metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available, skipping W&B logging")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(original_data_path, cluster_assignments_path):
    """
    Load original data and cluster assignments.
    
    Returns:
        df: Merged dataframe with cluster assignments
    """
    print("Loading data...")
    
    # Load original data
    df_original = pd.read_csv(original_data_path)
    
    # Load cluster assignments
    df_clusters = pd.read_csv(cluster_assignments_path)
    
    # Merge
    df = df_original.merge(
        df_clusters[['game_id', 'profile_id', 'cluster']], 
        on=['game_id', 'profile_id'],
        how='inner'
    )
    
    print(f"Loaded {len(df)} rows across {df['cluster'].nunique()} clusters")
    
    return df

# =============================================================================
# CLUSTER PROFILING
# =============================================================================

def profile_clusters(df, numerical_cols, categorical_cols, top_n_events=10, top_n_entities=5):
    """
    Generate comprehensive cluster profiles showing key characteristics.
    
    Returns:
        profiles: Dictionary of DataFrames with cluster statistics
        wandb_metrics: Dictionary of metrics for W&B logging
    """
    print("\n" + "="*70)
    print("CLUSTER PROFILING")
    print("="*70)
    
    profiles = {}
    wandb_metrics = {}
    
    # 1. NUMERICAL FEATURE STATISTICS
    print("\n1. Average Resource Metrics by Cluster")
    print("-" * 70)
    
    numerical_stats = df.groupby('cluster')[numerical_cols].mean()
    profiles['numerical'] = numerical_stats
    
    # Log to W&B
    for cluster in numerical_stats.index:
        for col in numerical_cols:
            wandb_metrics[f'cluster_{cluster}/{col}_mean'] = numerical_stats.loc[cluster, col]
    
    print(numerical_stats.round(2))
    
    # 2. PHASE DISTRIBUTION
    print("\n2. Phase Distribution by Cluster (%)")
    print("-" * 70)
    
    phase_dist = pd.crosstab(
        df['cluster'], 
        df['phase'], 
        normalize='index'
    ) * 100
    
    profiles['phase_distribution'] = phase_dist
    
    # Log to W&B
    for cluster in phase_dist.index:
        for phase in phase_dist.columns:
            wandb_metrics[f'cluster_{cluster}/phase_{phase}_pct'] = phase_dist.loc[cluster, phase]
    
    print(phase_dist.round(1))
    
    # 3. TOP EVENTS PER CLUSTER
    print(f"\n3. Top {top_n_events} Events by Cluster")
    print("-" * 70)
    
    event_profiles = {}
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        top_events = cluster_df['event'].value_counts().head(top_n_events)
        event_profiles[f'Cluster_{cluster}'] = top_events
        
        print(f"\nCluster {cluster}:")
        for event, count in top_events.items():
            pct = 100 * count / len(cluster_df)
            print(f"  {event}: {count} ({pct:.1f}%)")
            wandb_metrics[f'cluster_{cluster}/event_{event}_pct'] = pct
    
    profiles['top_events'] = pd.DataFrame(event_profiles)
    
    # 4. TOP ENTITIES PER CLUSTER
    print(f"\n4. Top {top_n_entities} Entities by Cluster")
    print("-" * 70)
    
    entity_profiles = {}
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        top_entities = cluster_df['entity'].value_counts().head(top_n_entities)
        entity_profiles[f'Cluster_{cluster}'] = top_entities
        
        print(f"\nCluster {cluster}:")
        for entity, count in top_entities.items():
            pct = 100 * count / len(cluster_df)
            print(f"  {entity}: {count} ({pct:.1f}%)")
            wandb_metrics[f'cluster_{cluster}/entity_{entity}_pct'] = pct
    
    profiles['top_entities'] = pd.DataFrame(entity_profiles)
    
    # 5. CIVILIZATION PREFERENCES
    print("\n5. Player Civilization Distribution by Cluster (%)")
    print("-" * 70)
    
    civ_dist = pd.crosstab(
        df.groupby(['cluster', 'game_id', 'profile_id'])['player_civ'].first().reset_index()['cluster'],
        df.groupby(['cluster', 'game_id', 'profile_id'])['player_civ'].first().reset_index()['player_civ'],
        normalize='index'
    ) * 100
    
    profiles['civ_distribution'] = civ_dist
    
    # Log top civs per cluster to W&B
    for cluster in civ_dist.index:
        top_3_civs = civ_dist.loc[cluster].nlargest(3)
        for i, (civ, pct) in enumerate(top_3_civs.items()):
            wandb_metrics[f'cluster_{cluster}/top_civ_{i+1}'] = f"{civ} ({pct:.1f}%)"
    
    print(civ_dist.round(1))
    
    return profiles, wandb_metrics

# =============================================================================
# TEMPORAL PATTERN ANALYSIS
# =============================================================================

def analyze_temporal_patterns(df, numerical_cols):
    """
    Analyze temporal patterns: timing, sequences, and progression.
    
    Returns:
        temporal_stats: Dictionary with temporal analysis results
        wandb_metrics: Dictionary of metrics for W&B logging
    """
    print("\n" + "="*70)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*70)
    
    temporal_stats = {}
    wandb_metrics = {}
    
    # 1. GAME DURATION BY CLUSTER
    print("\n1. Average Game Duration by Cluster")
    print("-" * 70)
    
    duration_stats = df.groupby(['cluster', 'game_id', 'profile_id'])['time'].max().reset_index()
    duration_by_cluster = duration_stats.groupby('cluster')['time'].agg(['mean', 'median', 'std'])
    
    temporal_stats['game_duration'] = duration_by_cluster
    
    # Log to W&B
    for cluster in duration_by_cluster.index:
        wandb_metrics[f'cluster_{cluster}/avg_game_duration'] = duration_by_cluster.loc[cluster, 'mean']
        wandb_metrics[f'cluster_{cluster}/median_game_duration'] = duration_by_cluster.loc[cluster, 'median']
    
    print(duration_by_cluster.round(2))
    
    # 2. EARLY GAME CHARACTERISTICS (first 10% of time)
    print("\n2. Early Game Characteristics (First 10% of Game Time)")
    print("-" * 70)
    
    early_game_data = []
    for (cluster, game_id, profile_id), group in df.groupby(['cluster', 'game_id', 'profile_id']):
        max_time = group['time'].max()
        early_cutoff = max_time * 0.1
        early_game = group[group['time'] <= early_cutoff]
        
        if len(early_game) > 0:
            early_game_data.append({
                'cluster': cluster,
                'early_events': len(early_game),
                'early_wood_rate': early_game['wood_per_min'].mean() if 'wood_per_min' in early_game.columns else 0,
                'early_food_rate': early_game['food_per_min'].mean() if 'food_per_min' in early_game.columns else 0,
                'most_common_event': early_game['event'].mode()[0] if len(early_game['event'].mode()) > 0 else None
            })
    
    early_df = pd.DataFrame(early_game_data)
    early_stats = early_df.groupby('cluster')[['early_events', 'early_wood_rate', 'early_food_rate']].mean()
    
    temporal_stats['early_game'] = early_stats
    
    # Log to W&B
    for cluster in early_stats.index:
        wandb_metrics[f'cluster_{cluster}/early_wood_rate'] = early_stats.loc[cluster, 'early_wood_rate']
        wandb_metrics[f'cluster_{cluster}/early_food_rate'] = early_stats.loc[cluster, 'early_food_rate']
    
    print(early_stats.round(2))
    
    # 3. PHASE TRANSITION TIMING
    print("\n3. Phase Transition Timing")
    print("-" * 70)
    
    phase_transitions = []
    for (cluster, game_id, profile_id), group in df.groupby(['cluster', 'game_id', 'profile_id']):
        group_sorted = group.sort_values('time')
        
        # Find when each phase first appears
        for phase in group_sorted['phase'].unique():
            first_occurrence = group_sorted[group_sorted['phase'] == phase]['time'].min()
            phase_transitions.append({
                'cluster': cluster,
                'phase': phase,
                'time': first_occurrence
            })
    
    phase_df = pd.DataFrame(phase_transitions)
    phase_timing = phase_df.groupby(['cluster', 'phase'])['time'].mean().unstack(fill_value=np.nan)
    
    temporal_stats['phase_timing'] = phase_timing
    
    # Log to W&B
    for cluster in phase_timing.index:
        for phase in phase_timing.columns:
            if not pd.isna(phase_timing.loc[cluster, phase]):
                wandb_metrics[f'cluster_{cluster}/phase_{phase}_timing'] = phase_timing.loc[cluster, phase]
    
    print(phase_timing.round(2))
    
    return temporal_stats, wandb_metrics

# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

def statistical_validation(df, numerical_cols, categorical_cols):
    """
    Perform statistical tests to validate cluster differences.
    
    Returns:
        validation_results: Dictionary with statistical test results
        wandb_metrics: Dictionary of metrics for W&B logging
    """
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION")
    print("="*70)
    
    validation_results = {}
    wandb_metrics = {}
    
    # 1. ANOVA FOR NUMERICAL FEATURES
    print("\n1. ANOVA Test: Do clusters differ significantly in numerical features?")
    print("-" * 70)
    
    anova_results = []
    for col in numerical_cols:
        # Get data for each cluster
        cluster_groups = [df[df['cluster'] == c][col].dropna() for c in sorted(df['cluster'].unique())]
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*cluster_groups)
        
        anova_results.append({
            'feature': col,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': 'YES' if p_value < 0.05 else 'NO'
        })
        
        wandb_metrics[f'anova/{col}_f_stat'] = f_stat
        wandb_metrics[f'anova/{col}_p_value'] = p_value
        
        sig_stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        print(f"{col:20s} F={f_stat:8.2f}, p={p_value:.4f} {sig_stars}")
    
    validation_results['anova'] = pd.DataFrame(anova_results)
    
    # 2. CHI-SQUARE FOR CATEGORICAL FEATURES
    print("\n2. Chi-Square Test: Are categorical features distributed differently?")
    print("-" * 70)
    
    chi_square_results = []
    for col in categorical_cols:
        try:
            # Create contingency table
            contingency = pd.crosstab(df['cluster'], df[col])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            chi_square_results.append({
                'feature': col,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': 'YES' if p_value < 0.05 else 'NO'
            })
            
            wandb_metrics[f'chi_square/{col}_chi2'] = chi2
            wandb_metrics[f'chi_square/{col}_p_value'] = p_value
            
            sig_stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            print(f"{col:20s} χ²={chi2:8.2f}, p={p_value:.4f} {sig_stars}")
        except Exception as e:
            print(f"{col:20s} Error: {str(e)}")
    
    validation_results['chi_square'] = pd.DataFrame(chi_square_results)
    
    # 3. CLUSTER SEPARATION QUALITY
    print("\n3. Cluster Quality Metrics")
    print("-" * 70)
    
    # Count games per cluster
    games_per_cluster = df.groupby('cluster')[['game_id', 'profile_id']].apply(
        lambda x: x.drop_duplicates().shape[0]
    )
    
    print(f"Games per cluster:")
    for cluster, count in games_per_cluster.items():
        pct = 100 * count / games_per_cluster.sum()
        print(f"  Cluster {cluster}: {count} games ({pct:.1f}%)")
        wandb_metrics[f'cluster_quality/cluster_{cluster}_size'] = count
        wandb_metrics[f'cluster_quality/cluster_{cluster}_pct'] = pct
    
    validation_results['cluster_sizes'] = games_per_cluster
    
    return validation_results, wandb_metrics

# =============================================================================
# REPRESENTATIVE GAME SELECTION
# =============================================================================

def select_representative_games(df, n_per_cluster=5):
    """
    Select representative games from each cluster for manual inspection.
    
    Returns:
        representative_games: DataFrame with game IDs closest to cluster centers
    """
    print("\n" + "="*70)
    print("REPRESENTATIVE GAME SELECTION")
    print("="*70)
    print(f"\nSelecting {n_per_cluster} representative games per cluster for manual review...")
    
    # Load latent embeddings
    embeddings_df = pd.read_csv("outputs/latent_embeddings.csv")
    
    representative_games = []
    
    for cluster in sorted(embeddings_df['cluster'].unique()):
        cluster_data = embeddings_df[embeddings_df['cluster'] == cluster]
        
        # Get latent vectors
        latent_cols = [col for col in cluster_data.columns if col.startswith('latent_')]
        latent_vectors = cluster_data[latent_cols].values
        
        # Compute cluster center
        center = latent_vectors.mean(axis=0)
        
        # Find games closest to center
        distances = np.linalg.norm(latent_vectors - center, axis=1)
        closest_indices = np.argsort(distances)[:n_per_cluster]
        
        for idx in closest_indices:
            game_idx = cluster_data.iloc[idx]
            representative_games.append({
                'cluster': cluster,
                'game_id': game_idx['game_id'],
                'profile_id': game_idx['profile_id'],
                'distance_from_center': distances[idx]
            })
    
    rep_df = pd.DataFrame(representative_games)
    
    print("\nRepresentative games (closest to cluster centers):")
    for cluster in sorted(rep_df['cluster'].unique()):
        print(f"\nCluster {cluster}:")
        cluster_games = rep_df[rep_df['cluster'] == cluster]
        for _, row in cluster_games.iterrows():
            print(f"  Game {row['game_id']}, Player {row['profile_id']} (distance: {row['distance_from_center']:.4f})")
    
    return rep_df

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df, profiles, temporal_stats, numerical_cols, output_dir):
    """
    Create visualization plots for cluster analysis.
    
    Returns:
        wandb_images: Dictionary of images for W&B logging
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    wandb_images = {}
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Resource metrics by cluster
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        if idx < len(axes):
            ax = axes[idx]
            df.groupby('cluster')[col].mean().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(f'Average {col} by Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=0)
    
    # Hide unused subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filepath = f"{output_dir}/numerical_features_by_cluster.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    wandb_images['numerical_features'] = wandb.Image(filepath) if WANDB_AVAILABLE else None
    print(f"Saved: numerical_features_by_cluster.png")
    plt.close()
    
    # 2. Phase distribution heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(profiles['phase_distribution'], annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Phase Distribution by Cluster (%)')
    plt.xlabel('Phase')
    plt.ylabel('Cluster')
    plt.tight_layout()
    filepath = f"{output_dir}/phase_distribution_heatmap.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    wandb_images['phase_distribution'] = wandb.Image(filepath) if WANDB_AVAILABLE else None
    print(f"Saved: phase_distribution_heatmap.png")
    plt.close()
    
    # 3. Game duration by cluster
    plt.figure(figsize=(10, 6))
    duration_data = df.groupby(['cluster', 'game_id', 'profile_id'])['time'].max().reset_index()
    sns.boxplot(data=duration_data, x='cluster', y='time', palette='Set2')
    plt.title('Game Duration Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Game Duration (time units)')
    plt.tight_layout()
    filepath = f"{output_dir}/game_duration_by_cluster.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    wandb_images['game_duration'] = wandb.Image(filepath) if WANDB_AVAILABLE else None
    print(f"Saved: game_duration_by_cluster.png")
    plt.close()
    
    # 4. Cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_sizes = df.groupby('cluster')[['game_id', 'profile_id']].apply(
        lambda x: x.drop_duplicates().shape[0]
    )
    cluster_sizes.plot(kind='bar', color='coral')
    plt.title('Number of Games per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=0)
    plt.tight_layout()
    filepath = f"{output_dir}/cluster_sizes.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    wandb_images['cluster_sizes'] = wandb.Image(filepath) if WANDB_AVAILABLE else None
    print(f"Saved: cluster_sizes.png")
    plt.close()
    
    return wandb_images

# =============================================================================
# MAIN ANALYSIS FUNCTION (CALLABLE FROM TRAINING SCRIPT)
# =============================================================================

def run_analysis_and_log_to_wandb(original_data_path, cluster_assignments_path, 
                                   numerical_cols, categorical_cols, 
                                   output_dir="cluster_analysis", use_wandb=True):
    """
    Run complete cluster analysis and log results to W&B.
    
    This function can be called from the training script to automatically
    run analysis after clustering.
    
    Returns:
        analysis_results: Dictionary with all analysis results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(original_data_path, cluster_assignments_path)
    
    # Initialize W&B metrics dictionary
    all_wandb_metrics = {}
    
    # Profile clusters
    profiles, profile_metrics = profile_clusters(df, numerical_cols, categorical_cols)
    all_wandb_metrics.update(profile_metrics)
    
    # Analyze temporal patterns
    temporal_stats, temporal_metrics = analyze_temporal_patterns(df, numerical_cols)
    all_wandb_metrics.update(temporal_metrics)
    
    # Statistical validation
    validation_results, validation_metrics = statistical_validation(df, numerical_cols, categorical_cols)
    all_wandb_metrics.update(validation_metrics)
    
    # Create visualizations
    wandb_images = create_visualizations(df, profiles, temporal_stats, numerical_cols, output_dir)
    
    # Log everything to W&B
    if use_wandb and WANDB_AVAILABLE:
        print("\nLogging metrics to W&B...")
        wandb.log(all_wandb_metrics)
        
        # Log images
        for name, image in wandb_images.items():
            if image is not None:
                wandb.log({f"visualizations/{name}": image})
        
        print("W&B logging complete!")
    
    # Save all results to CSV
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    profiles['numerical'].to_csv(f"{output_dir}/cluster_numerical_profiles.csv")
    profiles['phase_distribution'].to_csv(f"{output_dir}/cluster_phase_distribution.csv")
    profiles['top_events'].to_csv(f"{output_dir}/cluster_top_events.csv")
    profiles['top_entities'].to_csv(f"{output_dir}/cluster_top_entities.csv")
    temporal_stats['game_duration'].to_csv(f"{output_dir}/cluster_game_duration.csv")
    temporal_stats['early_game'].to_csv(f"{output_dir}/cluster_early_game_stats.csv")
    validation_results['anova'].to_csv(f"{output_dir}/anova_results.csv", index=False)
    validation_results['chi_square'].to_csv(f"{output_dir}/chi_square_results.csv", index=False)
    
    print(f"\nAll results saved to {output_dir}/")
    
    return {
        'profiles': profiles,
        'temporal_stats': temporal_stats,
        'validation_results': validation_results,
        'wandb_metrics': all_wandb_metrics
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def main():
    """Run complete cluster analysis pipeline as standalone script."""
    
    # Configuration
    original_data_path = "transformer_input_new.csv"
    cluster_assignments_path = "outputs/cluster_assignments.csv"
    output_dir = "cluster_analysis"
    
    numerical_cols = ["time", "wood", "stone", "food", "gold", 
                     "wood_per_min", "stone_per_min", "food_per_min", "gold_per_min"]
    categorical_cols = ["phase", "event", "entity", "player_civ", "enemy_civ", "map"]
    
    # Run analysis
    analysis_results = run_analysis_and_log_to_wandb(
        original_data_path=original_data_path,
        cluster_assignments_path=cluster_assignments_path,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        output_dir=output_dir,
        use_wandb=True  # Set to True if running standalone with W&B
    )
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR STRATEGY VALIDATION")
    print("="*70)
    print("""
1. MANUAL REVIEW:
   - Examine games from different clusters in detail
   - Watch replays or review detailed logs
   - Confirm if the patterns match real strategies
   
2. DOMAIN EXPERT VALIDATION:
   - Share the cluster profiles with game experts
   - Ask them to name the strategies based on characteristics
   - Compare their names with the patterns you see
   
3. OUTCOME ANALYSIS (if you have win/loss data):
   - Add a 'outcome' column to your CSV (win/loss)
   - Re-run analysis to see which strategies win most often
   - This validates if clusters capture meaningful strategic choices
   
4. REFINE CLUSTERS:
   - If clusters don't make strategic sense, try different n_clusters
   - Experiment with 3, 5, 7, 10 clusters
   - Look for the "sweet spot" where clusters are distinct and interpretable
   
5. CHECK W&B DASHBOARD:
   - Review all metrics and visualizations in W&B
   - Compare different runs with different hyperparameters
   - Share findings with your team
    """)


if __name__ == "__main__":
    main()
