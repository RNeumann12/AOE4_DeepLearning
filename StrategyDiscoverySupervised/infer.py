import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from StrategyDiscoverySupervised.dataset_resource_based import GameSequenceDataset2
from model import StrategyGRU
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from tqdm import tqdm


def load_model(model_path, dataset, device):
    """Load trained model from checkpoint"""
    model = StrategyGRU(
        num_numeric=len(dataset.NUMERICAL_COLS),
        num_events=len(dataset.event_vocab),
        hidden_size=128,
        num_layers=2,
        num_classes=len(dataset.strat_vocab),
        dropout=0.35,
        num_civs=len(dataset.civ_vocab),
        num_enemy_civs=len(dataset.enemy_civ_vocab),
        num_maps=len(dataset.map_vocab),
        num_ages=len(dataset.age_vocab) if dataset.age_vocab else 1
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✓ Model loaded from {model_path}")
    return model


@torch.no_grad()
def predict_batch(model, numeric_seq, event_seq, time_seq, mask, player_civ, enemy_civ, map_id, age, device):
    """Make predictions on a batch"""
    numeric_seq = numeric_seq.to(device)
    event_seq = event_seq.to(device)
    time_seq = time_seq.to(device)
    mask = mask.to(device)
    player_civ = player_civ.to(device)
    enemy_civ = enemy_civ.to(device)
    map_id = map_id.to(device)
    age = age.to(device)
    
    logits, _ = model(numeric_seq, event_seq, mask, time_seq, player_civ, enemy_civ, map_id, age)
    
    # Get predictions and confidences
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidences, preds = torch.max(probs, dim=-1)
    
    return preds.cpu().numpy(), confidences.cpu().numpy(), logits.cpu().numpy()


def test_on_dataset(model_path, csv_path, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=128):
    """
    Test model on a dataset
    
    Args:
        model_path: Path to saved model checkpoint
        csv_path: Path to test CSV file (same format as training)
        device: Device to run on
        batch_size: Batch size for inference
    """
    print(f"\n{'='*60}")
    print(f"Testing model on: {csv_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset = GameSequenceDataset2(csv_path, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = load_model(model_path, dataset, device)
    model.eval()
    
    # Create reverse mapping
    idx_to_strat = {v: k for k, v in dataset.strat_vocab.items()}
    
    # Collect predictions
    all_preds = []
    all_targets = []
    all_confidences = []
    all_logits = []
    
    print("Running inference...")
    for numeric_seq, event_seq, time_seq, mask, player_civ, enemy_civ, map_id, age, targets in tqdm(dataloader):
        preds, confidences, logits = predict_batch(
            model, numeric_seq, event_seq, time_seq, mask, 
            player_civ, enemy_civ, map_id, age, device
        )
        
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        all_confidences.extend(confidences)
        all_logits.extend(logits)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    all_logits = np.array(all_logits)
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {all_confidences.mean():.4f}")
    print(f"Median Confidence: {np.median(all_confidences):.4f}")
    
    # Per-class metrics
    print(f"\n{'Strategy Name':<30} {'Accuracy':<12} {'Count':<12}")
    print("=" * 54)
    
    for class_id in range(len(dataset.strat_vocab)):
        class_name = idx_to_strat[class_id]
        class_mask = all_targets == class_id
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == class_id).sum() / class_mask.sum()
            class_count = class_mask.sum()
            print(f"{class_name:<30} {class_acc:<12.4f} {class_count:<12}")
    
    # Confusion Matrix
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*60}\n")
    
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(dataset.strat_vocab))))
    
    # Print formatted confusion matrix
    print(f"{'':30}", end='')
    for class_id in range(len(dataset.strat_vocab)):
        print(f"{idx_to_strat[class_id]:<15}", end='')
    print()
    
    for class_id in range(len(dataset.strat_vocab)):
        print(f"{idx_to_strat[class_id]:<30}", end='')
        for pred_id in range(len(dataset.strat_vocab)):
            print(f"{cm[class_id, pred_id]:<15}", end='')
        print()
    
    # Classification report
    print(f"\n{'='*60}")
    print(f"DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}\n")
    
    target_names = [idx_to_strat[i] for i in range(len(dataset.strat_vocab))]
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'True Strategy': [idx_to_strat[t] for t in all_targets],
        'Predicted Strategy': [idx_to_strat[p] for p in all_preds],
        'Confidence': all_confidences,
        'Correct': all_preds == all_targets
    })
    
    results_df.to_csv('test_results.csv', index=False)
    print(f"\n✓ Results saved to test_results.csv")
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'confidences': all_confidences,
        'confusion_matrix': cm,
        'results_df': results_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test strategy prediction model')
    parser.add_argument('--model', type=str, default='best_strategy_model.pt', help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, default='transformer_input_test_v2.csv', help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    results = test_on_dataset(
        model_path=args.model,
        csv_path=args.csv,
        device=args.device,
        batch_size=args.batch_size
    )
