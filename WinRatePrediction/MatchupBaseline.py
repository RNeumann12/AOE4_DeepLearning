"""
Baseline model for win rate prediction using only matchup features (player_civ, enemy_civ, map).

This serves to validate how much predictive power comes from civilization matchups vs actual gameplay.
Uses Logistic Regression (linear classifier for binary classification).

Usage:
    python WinRatePrediction/MatchupBaseline.py --csv transformer_input.csv
    python WinRatePrediction/MatchupBaseline.py --csv transformer_input.csv --test_split 0.2
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss
)


class MatchupBaselineModel:
    """
    A simple logistic regression baseline that predicts win/loss 
    based only on player_civ, enemy_civ, and map.
    
    This helps determine how much of the transformer model's performance
    comes from matchup statistics vs actual sequence learning.
    """
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False
        
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode matchup features from dataframe."""
        # Get unique player-game combinations (one row per player per game)
        games = df.groupby(['game_id', 'profile_id']).first().reset_index()
        
        # Extract matchup features
        features = games[['player_civ', 'enemy_civ', 'map']].fillna('unknown').astype(str)
        labels = games['player_won'].values
        
        return features, labels, games
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the baseline model on training data.
        
        Args:
            df: DataFrame with columns player_civ, enemy_civ, map, player_won
        """
        features, labels, _ = self._prepare_features(df)
        
        # Fit encoder and transform
        X = self.encoder.fit_transform(features)
        
        # Fit logistic regression
        self.model.fit(X, labels)
        self.is_fitted = True
        
        print(f"Fitted on {len(labels)} player-game samples")
        print(f"Feature dimensions: {X.shape[1]} (one-hot encoded)")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        features, _, games = self._prepare_features(df)
        X = self.encoder.transform(features)
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_binary(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary win/loss."""
        probs = self.predict(df)
        return (probs >= 0.5).astype(int)
    
    def evaluate(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            df: DataFrame to evaluate on
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of metrics
        """
        features, labels, _ = self._prepare_features(df)
        X = self.encoder.transform(features)
        
        # Get predictions
        probs = self.model.predict_proba(X)[:, 1]
        preds_bin = (probs >= 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(labels, preds_bin)
        bal_acc = balanced_accuracy_score(labels, preds_bin)
        
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float('nan')
        
        brier = brier_score_loss(labels, probs)
        
        tn, fp, fn, tp = confusion_matrix(labels, preds_bin).ravel()
        
        precision_win = tp / (tp + fp + 1e-8)
        precision_loss = tn / (tn + fn + 1e-8)
        recall_win = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision_win * recall_win) / (precision_win + recall_win + 1e-8)
        
        metrics = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'auc': auc,
            'brier': brier,
            'f1': f1,
            'precision_win': precision_win,
            'precision_loss': precision_loss,
            'recall_win': recall_win,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'n_samples': len(labels),
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("MATCHUP BASELINE MODEL EVALUATION")
            print("(Logistic Regression on player_civ, enemy_civ, map only)")
            print("=" * 60)
            print(f"\n--- Core Metrics ---")
            print(f"  Accuracy:          {acc:.4f}")
            print(f"  Balanced Accuracy: {bal_acc:.4f}")
            print(f"  AUC-ROC:           {auc:.4f}")
            print(f"  F1 Score:          {f1:.4f}")
            print(f"  Brier Score:       {brier:.4f}")
            
            print(f"\n--- Confusion Matrix ---")
            print(f"  True Positives (wins correctly predicted):   {tp}")
            print(f"  True Negatives (losses correctly predicted): {tn}")
            print(f"  False Positives (predicted win, was loss):   {fp}")
            print(f"  False Negatives (predicted loss, was win):   {fn}")
            
            print(f"\n--- Precision / Recall ---")
            print(f"  Win Precision:  {precision_win:.4f}")
            print(f"  Loss Precision: {precision_loss:.4f}")
            print(f"  Win Recall:     {recall_win:.4f}")
            print("=" * 60 + "\n")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance (coefficient magnitudes) from logistic regression.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        feature_names = self.encoder.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_importance': np.abs(coefficients)
        }).sort_values('abs_importance', ascending=False)
        
        return importance_df.head(top_n)


class MatchupLookupBaseline:
    """
    Even simpler baseline: just look up historical win rate for each matchup.
    No machine learning, just statistics.
    """
    
    def __init__(self):
        self.matchup_rates = None
        self.overall_rate = 0.5
        
    def fit(self, df: pd.DataFrame):
        """Calculate historical win rates per matchup."""
        games = df.groupby(['game_id', 'profile_id']).first().reset_index()
        
        # Calculate win rate per matchup
        self.matchup_rates = games.groupby(
            ['player_civ', 'enemy_civ', 'map']
        )['player_won'].mean().to_dict()
        
        # Fallback to civ-only matchup
        self.civ_rates = games.groupby(
            ['player_civ', 'enemy_civ']
        )['player_won'].mean().to_dict()
        
        # Overall fallback
        self.overall_rate = games['player_won'].mean()
        
        print(f"Fitted lookup table with {len(self.matchup_rates)} matchup combinations")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict based on historical matchup win rates."""
        games = df.groupby(['game_id', 'profile_id']).first().reset_index()
        
        probs = []
        for _, row in games.iterrows():
            key = (row['player_civ'], row['enemy_civ'], row['map'])
            if key in self.matchup_rates:
                probs.append(self.matchup_rates[key])
            else:
                # Fallback to civ-only
                civ_key = (row['player_civ'], row['enemy_civ'])
                if civ_key in self.civ_rates:
                    probs.append(self.civ_rates[civ_key])
                else:
                    probs.append(self.overall_rate)
        
        return np.array(probs)
    
    def evaluate(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """Evaluate the lookup baseline."""
        games = df.groupby(['game_id', 'profile_id']).first().reset_index()
        labels = games['player_won'].values
        
        probs = self.predict(df)
        preds_bin = (probs >= 0.5).astype(int)
        
        acc = accuracy_score(labels, preds_bin)
        bal_acc = balanced_accuracy_score(labels, preds_bin)
        
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float('nan')
        
        metrics = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'auc': auc,
            'n_samples': len(labels),
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("MATCHUP LOOKUP BASELINE (Historical Win Rates)")
            print("=" * 60)
            print(f"  Accuracy:          {acc:.4f}")
            print(f"  Balanced Accuracy: {bal_acc:.4f}")
            print(f"  AUC-ROC:           {auc:.4f}")
            print("=" * 60 + "\n")
        
        return metrics


def main(args):
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Get unique games for splitting
    games = df.groupby(['game_id', 'profile_id']).first().reset_index()
    game_ids = games['game_id'].unique()
    
    # Split by game_id (same as transformer training)
    train_games, test_games = train_test_split(
        game_ids, test_size=args.test_split, random_state=42
    )
    
    train_df = df[df['game_id'].isin(train_games)]
    test_df = df[df['game_id'].isin(test_games)]
    
    print(f"Train games: {len(train_games)}, Test games: {len(test_games)}")
    
    # ===== Logistic Regression Baseline =====
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    
    lr_model = MatchupBaselineModel()
    lr_model.fit(train_df)
    
    print("\n--- Training Set Performance ---")
    train_metrics = lr_model.evaluate(train_df, verbose=True)
    
    print("\n--- Test Set Performance ---")
    test_metrics = lr_model.evaluate(test_df, verbose=True)
    
    print("\n--- Top Feature Importances ---")
    importance = lr_model.get_feature_importance(top_n=15)
    print(importance.to_string(index=False))
    
    # ===== Lookup Baseline =====
    print("\n" + "=" * 60)
    print("TRAINING LOOKUP BASELINE (Pure Statistics)")
    print("=" * 60)
    
    lookup_model = MatchupLookupBaseline()
    lookup_model.fit(train_df)
    
    print("\n--- Training Set Performance ---")
    lookup_train = lookup_model.evaluate(train_df, verbose=True)
    
    print("\n--- Test Set Performance ---")
    lookup_test = lookup_model.evaluate(test_df, verbose=True)
    
    # ===== Summary Comparison =====
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<30} {'Train Acc':<12} {'Test Acc':<12} {'Test AUC':<12}")
    print("-" * 66)
    print(f"{'Logistic Regression':<30} {train_metrics['accuracy']:.4f}       {test_metrics['accuracy']:.4f}       {test_metrics['auc']:.4f}")
    print(f"{'Lookup (Historical Rates)':<30} {lookup_train['accuracy']:.4f}       {lookup_test['accuracy']:.4f}       {lookup_test['auc']:.4f}")
    print(f"{'Random Baseline':<30} {'0.5000':<12} {'0.5000':<12} {'0.5000':<12}")
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print(f"""
If the Transformer model with sequence length N achieves X% accuracy,
compare to these baselines:

- Logistic Regression (matchup only): {test_metrics['accuracy']*100:.1f}%
- Lookup Baseline (matchup only):     {lookup_test['accuracy']*100:.1f}%
- Random:                              50.0%

The TRUE contribution of sequences = Transformer accuracy - Matchup baseline
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Matchup-based baseline for win rate prediction'
    )
    parser.add_argument('--csv', type=str, default='transformer_input.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of data to use for testing')
    args = parser.parse_args()
    main(args)
