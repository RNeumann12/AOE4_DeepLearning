"""
Inference script: load saved model and run on dataset, printing / saving predictions.

Usage examples:
Standard inference over games to predict if player will win:
    python WinRatePrediction/WinRate_infer.py --model best_model_len_200_no_destroy.pt --csv transformer_input_new.csv --max_len 200 --filter_destroy_events  --print_game_table

Sweep over sequence lengths to analyze accuracy vs length:
    python WinRatePrediction/WinRate_infer.py --model best_model_len_200_no_destroy.pt --csv transformer_input_new.csv --sweep_lengths --max_len 200 --length_step 50 --filter_destroy_events

Validation on dataset:
  python WinRatePrediction/WinRate_infer.py --model WinRatePrediction/winrate_final_model.pt --csv training_data_2026_01.csv --max_len 100 --filter_destroy_events

   """
import argparse
import tempfile
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, brier_score_loss
# Ensure project root is on PYTHONPATH for local imports
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aoe_player_game_datset import AoEEventDataset, collate_fn
from WinRatePrediction.WinRateTransformerModel import AoETransformer
from torch.utils.data import DataLoader


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += np.abs(bin_conf - bin_acc) * (mask.sum() / len(probs))

    return ece


def compute_metrics(preds, trues):
    """Compute and print evaluation metrics."""
    preds = np.array(preds)
    trues = np.array(trues)

    # AUC
    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = float('nan')

    # Binary predictions
    preds_bin = (preds >= 0.5).astype(int)
    acc = accuracy_score(trues, preds_bin)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(trues, preds_bin).ravel()

    # Rates
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    wprecision = tp / (tp + fp + 1e-8)
    lprecision = tn / (tn + fn + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    balanced_acc = 0.5 * (tp / (tp + fn + 1e-8) + tn / (tn + fp + 1e-8))
    f1score = 2 * ((wprecision * recall) / (wprecision + recall + 1e-8))

    # Calibration metrics
    ece = expected_calibration_error(preds, trues)
    brier = brier_score_loss(trues, preds)

    # Confidence analysis
    CONF_HIGH = 0.7
    CONF_LOW = 0.3
    confident_wrong_win = preds[(preds >= CONF_HIGH) & (trues == 0)]
    confident_wrong_loss = preds[(preds <= CONF_LOW) & (trues == 1)]

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"\n--- Core Metrics ---")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  F1 Score:          {f1score:.4f}")
    print(f"  Brier Score:       {brier:.4f}")
    print(f"  ECE:               {ece:.4f}")

    print(f"\n--- Confusion Matrix ---")
    print(f"  True Positives (wins correctly predicted):   {tp}")
    print(f"  True Negatives (losses correctly predicted): {tn}")
    print(f"  False Positives (predicted win, was loss):   {fp}")
    print(f"  False Negatives (predicted loss, was win):   {fn}")

    print(f"\n--- Precision / Recall ---")
    print(f"  Win Precision:  {wprecision:.4f}")
    print(f"  Loss Precision: {lprecision:.4f}")
    print(f"  Win Recall:     {recall:.4f}")

    print(f"\n--- Error Rates ---")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")

    print(f"\n--- Confidence Analysis ---")
    print(f"  Confident wrong win rate (pred >= {CONF_HIGH}, true=0):  {len(confident_wrong_win) / len(preds):.4f}")
    print(f"  Confident wrong loss rate (pred <= {CONF_LOW}, true=1): {len(confident_wrong_loss) / len(preds):.4f}")
    print("=" * 50 + "\n")

    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'auc': auc,
        'f1': f1score,
        'brier': brier,
        'ece': ece,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision_win': wprecision,
        'precision_loss': lprecision,
        'recall_win': recall,
        'fpr': fpr,
        'fnr': fnr,
    }


def run_inference_for_length(model, vocabs, csv_path, max_len, truncation_strategy, batch_size, device):
    """Run inference for a specific max_len and return predictions and labels."""
    ds = AoEEventDataset(csv_path, vocabs['entity_vocab'], vocabs['event_vocab'], 
                         vocabs['civ_vocab'], vocabs['map_vocab'], 
                         max_len=max_len, truncation_strategy=truncation_strategy)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            # move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            logits = model(
                batch['entity_ids'],
                batch['event_ids'],
                batch['times'],
                batch['attention_mask'],
                batch['map'],
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ']
            )
            probs = torch.sigmoid(logits)
            preds.extend(list(probs.cpu().numpy()))
            trues.extend(list(batch['labels'].cpu().numpy()))

    return preds, trues


def sweep_lengths_and_plot(args, model, vocabs, device, csv_path):
    """Sweep through different max_len values and plot accuracy vs sequence length.

    Requires real win/loss labels — call only when has_labels is True.
    The csv_path should already be preprocessed (filtered, dummy labels injected if needed)
    by the caller.
    """
    # Determine the maximum length to sweep to
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['game_id', 'profile_id']).size()
    dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0

    if args.max_len is not None:
        sweep_max = args.max_len
    else:
        sweep_max = min(dataset_longest, args.max_pos_embed_cap)

    # Generate length values to test (from length_step to sweep_max, in increments of length_step)
    lengths = list(range(args.length_step, sweep_max + 1, args.length_step))
    if lengths and lengths[-1] != sweep_max:
        lengths.append(sweep_max)

    print(f"\n{'='*60}")
    print(f"SWEEPING SEQUENCE LENGTHS: {args.length_step} to {sweep_max} (step={args.length_step})")
    print(f"{'='*60}\n")
    
    results = {
        'length': [],
        'accuracy': [],
        'balanced_accuracy': [],
        'auc': [],
        'f1': [],
        'brier': [],
        'ece': []
    }
    
    for length in lengths:
        print(f"\n--- Evaluating max_len = {length} ---")
        preds, trues = run_inference_for_length(
            model, vocabs, csv_path, length, 
            args.truncation_strategy, args.batch_size, device
        )
        
        # Compute metrics (silently)
        preds_arr = np.array(preds)
        trues_arr = np.array(trues)
        
        try:
            auc = roc_auc_score(trues_arr, preds_arr)
        except Exception:
            auc = float('nan')
        
        preds_bin = (preds_arr >= 0.5).astype(int)
        acc = accuracy_score(trues_arr, preds_bin)
        
        tn, fp, fn, tp = confusion_matrix(trues_arr, preds_bin).ravel()
        balanced_acc = 0.5 * (tp / (tp + fn + 1e-8) + tn / (tn + fp + 1e-8))
        
        wprecision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1score = 2 * ((wprecision * recall) / (wprecision + recall + 1e-8))
        
        ece = expected_calibration_error(preds_arr, trues_arr)
        brier = brier_score_loss(trues_arr, preds_arr)
        
        results['length'].append(length)
        results['accuracy'].append(acc)
        results['balanced_accuracy'].append(balanced_acc)
        results['auc'].append(auc)
        results['f1'].append(f1score)
        results['brier'].append(brier)
        results['ece'].append(ece)
        
        print(f"  Accuracy: {acc:.4f} | Balanced Acc: {balanced_acc:.4f} | AUC: {auc:.4f} | F1: {f1score:.4f}")
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance vs Sequence Length', fontsize=14, fontweight='bold')
    
    # Plot 1: Accuracy metrics
    ax1 = axes[0, 0]
    ax1.plot(results['length'], results['accuracy'], 'b-o', label='Accuracy', markersize=5)
    ax1.plot(results['length'], results['balanced_accuracy'], 'g-s', label='Balanced Accuracy', markersize=5)
    ax1.set_xlabel('Sequence Length (max_len)')
    ax1.set_ylabel('Score')
    ax1.set_title('Accuracy Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: AUC and F1
    ax2 = axes[0, 1]
    ax2.plot(results['length'], results['auc'], 'r-o', label='AUC-ROC', markersize=5)
    ax2.plot(results['length'], results['f1'], 'm-s', label='F1 Score', markersize=5)
    ax2.set_xlabel('Sequence Length (max_len)')
    ax2.set_ylabel('Score')
    ax2.set_title('AUC-ROC and F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Calibration metrics (lower is better)
    ax3 = axes[1, 0]
    ax3.plot(results['length'], results['brier'], 'c-o', label='Brier Score', markersize=5)
    ax3.plot(results['length'], results['ece'], 'y-s', label='ECE', markersize=5)
    ax3.set_xlabel('Sequence Length (max_len)')
    ax3.set_ylabel('Score (lower is better)')
    ax3.set_title('Calibration Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table as text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find best length for each metric
    best_acc_idx = np.argmax(results['accuracy'])
    best_bal_idx = np.argmax(results['balanced_accuracy'])
    best_auc_idx = np.argmax(results['auc'])
    best_f1_idx = np.argmax(results['f1'])
    
    summary_text = (
        f"Summary Statistics\n"
        f"{'='*40}\n\n"
        f"Best Accuracy:          {results['accuracy'][best_acc_idx]:.4f} at len={results['length'][best_acc_idx]}\n"
        f"Best Balanced Accuracy: {results['balanced_accuracy'][best_bal_idx]:.4f} at len={results['length'][best_bal_idx]}\n"
        f"Best AUC-ROC:           {results['auc'][best_auc_idx]:.4f} at len={results['length'][best_auc_idx]}\n"
        f"Best F1 Score:          {results['f1'][best_f1_idx]:.4f} at len={results['length'][best_f1_idx]}\n\n"
        f"Length Range: {results['length'][0]} to {results['length'][-1]}\n"
        f"Step Size: {args.length_step}\n"
        f"Total Evaluations: {len(results['length'])}"
    )
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = args.sweep_plot_out if args.sweep_plot_out else 'length_sweep_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {plot_path}")
    
    # Also save results to CSV
    results_df = pd.DataFrame(results)
    csv_out = plot_path.replace('.png', '.csv')
    results_df.to_csv(csv_out, index=False)
    print(f"Results CSV saved to: {csv_out}")
    print(f"{'='*60}\n")

    plt.show()

    return results


def main(args):
    print('Loading saved model...')
    
    # Load checkpoint first to get the vocabs used during training
    chk = torch.load(args.model, map_location='cpu')
    vocabs = chk['vocabs']
    print(f"Loaded vocabs from checkpoint: entity={len(vocabs['entity_vocab'])}, event={len(vocabs['event_vocab'])}, civ={len(vocabs['civ_vocab'])}, map={len(vocabs['map_vocab'])}")

    # Get model max_len from the checkpoint's positional embedding shape
    model_max_len = chk['model_state']['seq_pos_embed.weight'].shape[0]
    print(f"Model positional embedding length from checkpoint: {model_max_len}")

    model = AoETransformer(vocab_size_entity=len(vocabs['entity_vocab']), vocab_size_event=len(vocabs['event_vocab']), civ_vocab_size=len(vocabs['civ_vocab']), map_vocab_size=len(vocabs['map_vocab']), d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.ffn_dim, dropout=args.dropout, max_len=model_max_len)
    model.load_state_dict(chk['model_state'])

    # Move model to selected device and set eval mode
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------ #
    # Load and preprocess CSV (shared by both sweep and standard modes)   #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(args.csv)

    # Optionally filter out DESTROY events
    if args.filter_destroy_events:
        original_len = len(df)
        df = df[df['event'] != 'DESTROY']
        print(f"Filtered out DESTROY events: {original_len} -> {len(df)} rows ({original_len - len(df)} removed)")

    # Detect whether real win/loss labels are available
    has_labels = 'player_won' in df.columns and df['player_won'].notna().any()
    if not has_labels:
        print("Note: 'player_won' column not found or empty — running in inference-only mode (metrics skipped).")
        df['player_won'] = 0  # dummy so AoEEventDataset doesn't crash

    # Write a temp CSV whenever the dataframe was modified (filtered or dummy label added)
    needs_temp = args.filter_destroy_events or not has_labels
    if needs_temp:
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_csv.name, index=False)
        csv_path = temp_csv.name
    else:
        csv_path = args.csv

    # If sweep mode is enabled, run the length sweep analysis (requires labels)
    if args.sweep_lengths:
        if not has_labels:
            print("Error: --sweep_lengths requires 'player_won' labels in the CSV (validation mode only).")
            if needs_temp:
                os.unlink(csv_path)
            return
        sweep_lengths_and_plot(args, model, vocabs, device, csv_path=csv_path)
        if needs_temp:
            os.unlink(csv_path)
        return

    # ------------------------------------------------------------------ #
    # Standard inference / validation pass                                #
    # ------------------------------------------------------------------ #
    grouped = df.groupby(['game_id', 'profile_id']).size()
    dataset_longest = int(grouped.max()) if len(grouped) > 0 else 0
    if args.max_len is not None:
        desired_max_len = args.max_len
    else:
        desired_max_len = min(dataset_longest, args.max_pos_embed_cap)
        if dataset_longest > desired_max_len:
            print(f"Warning: longest sequence in data is {dataset_longest}, capping positional length to {desired_max_len}. Sequences will be truncated per strategy '{args.truncation_strategy}'")

    ds = AoEEventDataset(csv_path, vocabs['entity_vocab'], vocabs['event_vocab'], vocabs['civ_vocab'], vocabs['map_vocab'], max_len=desired_max_len, truncation_strategy=args.truncation_strategy)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            logits = model(
                batch['entity_ids'],
                batch['event_ids'],
                batch['times'],
                batch['attention_mask'],
                batch['map'],
                player_civ=batch['player_civ'],
                enemy_civ=batch['enemy_civ']
            )
            probs = torch.sigmoid(logits)
            preds.extend(list(probs.cpu().numpy()))
            trues.extend(list(batch['labels'].cpu().numpy()))

    # Print basic stats
    print(f'Predictions: mean={np.mean(preds):.4f}, min={np.min(preds):.4f}, max={np.max(preds):.4f}')

    # Per-game probability table
    if args.print_game_table:
        rows = []
        for i, ex in enumerate(ds.examples):
            row = {
                'game_id':    ex['game_id'],
                'player_civ': ex['player_civ'],
                'enemy_civ':  ex['enemy_civ'],
                'map':        ex['map'],
                'win_prob':   round(float(preds[i]), 4),
            }
            if has_labels:
                row['actual_win'] = int(trues[i])
                row['correct'] = 'Y' if (preds[i] >= 0.5) == bool(trues[i]) else 'N'
            rows.append(row)
        game_df = pd.DataFrame(rows)
        print(f"\n{'='*80}")
        print(f"PER-GAME WIN PROBABILITY TABLE  (model={args.model}, max_len={desired_max_len}, truncation={args.truncation_strategy})")
        print(f"{'='*80}")
        print(game_df.to_string(index=False))
        print(f"{'='*80}\n")

    # Compute and print validation metrics (only when labels are available)
    if has_labels:
        compute_metrics(preds, trues)

    # Clean up temp file if created
    if needs_temp:
        os.unlink(csv_path)

    # Optionally save predictions
    if args.out:
        out_df = pd.DataFrame({'pred': preds})
        if has_labels:
            out_df['true'] = trues
        out_df.to_csv(args.out, index=False)
        print(f'Saved predictions to {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pt')
    parser.add_argument('--csv', type=str, default='transformer_input.csv')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=None, help='Positional embedding length; if omitted, use dataset max sequence length')
    parser.add_argument('--max_pos_embed_cap', type=int, default=1024, help='Safety cap for positional embeddings to avoid OOM when using very long sequences')
    parser.add_argument('--truncation_strategy', choices=['head','tail','head_tail'], default='head', help='How to truncate sequences when capped')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on ("cuda" or "cpu"). Will fall back to CPU if CUDA unavailable')
    parser.add_argument('--sweep_lengths', action='store_true', help='Sweep through different max_len values and plot accuracy')
    parser.add_argument('--length_step', type=int, default=5, help='Step size for length sweep (default: 5)')
    parser.add_argument('--sweep_plot_out', type=str, default='length_sweep_results.png', help='Output path for the length sweep plot')
    parser.add_argument('--filter_destroy_events', action='store_true', default=False, help='Filter out all DESTROY events from inference data')
    parser.add_argument('--print_game_table', action='store_true', default=False, help='Print a table showing win probability for every game (with model parameters)')
    args = parser.parse_args()
    main(args)
