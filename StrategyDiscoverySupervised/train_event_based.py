import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset_prep_event_based import DataSetPrep
from dataset_event_based import AoE4Dataset
from model_v2 import StrategyGRU
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb

BATCH_SIZE = 32
TEST_SIZE = 0.2
NUM_EPOCHS = 50
LR = 0.0001
PATIENCE = 5
CLASS_WEIGHT_METHODE = 'sqr'
WEIGHT_DECAY = 0.01

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(dataloader, desc="Epoch {epoch}Training", leave=False)
    for batch in pbar:
        sequence = batch['sequence'].to(device)
        mask = batch['mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(sequence, mask, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(dataloader)

    return train_acc, avg_train_loss

@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    all_predictions = []
    all_labels = []
    all_confidences = []
    false_classified_info = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequence = batch['sequence'].to(device)
            mask = batch['mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(sequence, mask, metadata)
            loss = criterion(outputs, labels)

            # probabilities
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)

            # calc loss
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Track false classifications
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    false_classified_info.append({
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'confidence': confidences[i].item(),
                        'all_probs': probs[i].cpu().numpy()
                    })
    
    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(dataloader)

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Precision, Recall, F1 (per class and averaged)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Per-class TP, FP, TN, FN and rates
    num_classes = cm.shape[0]
    class_metrics = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        total = tp + fp + tn + fn
        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        class_metrics.append({
            'class': i,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'tp_rate': tp_rate,
            'fp_rate': fp_rate,
            'tn_rate': tn_rate,
            'fn_rate': fn_rate,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        })
    
    metrics = {
        'accuracy': val_acc,
        'loss': avg_val_loss,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'false_classified': false_classified_info
    }
    
    return metrics


def log_metrics_to_wandb(metrics, label_names, epoch, phase='val'):
    """Log all metrics to wandb"""
    
    # Collect all metrics in one dict
    log_dict = {
        f"{phase}/accuracy": metrics['accuracy'],
        f"{phase}/loss": metrics['loss'],
        f"{phase}/precision_macro": metrics['precision_macro'],
        f"{phase}/recall_macro": metrics['recall_macro'],
        f"{phase}/f1_macro": metrics['f1_macro'],
    }
    
    # Per-class metrics
    for cm in metrics['class_metrics']:
        class_name = label_names[cm['class']] if label_names else f"class_{cm['class']}"
        log_dict.update({
            f"{phase}/{class_name}/precision": cm['precision'],
            f"{phase}/{class_name}/recall": cm['recall'],
            f"{phase}/{class_name}/f1": cm['f1'],
            f"{phase}/{class_name}/tp": cm['tp'],
            f"{phase}/{class_name}/fp": cm['fp'],
            f"{phase}/{class_name}/tn": cm['tn'],
            f"{phase}/{class_name}/fn": cm['fn'],
            f"{phase}/{class_name}/tp_rate": cm['tp_rate'],
            f"{phase}/{class_name}/fp_rate": cm['fp_rate'],
            f"{phase}/{class_name}/tn_rate": cm['tn_rate'],
            f"{phase}/{class_name}/fn_rate": cm['fn_rate'],
        })
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if label_names:
        display_labels = [label_names[i] for i in range(len(label_names))]
    else:
        display_labels = [f"Class {i}" for i in range(len(cm))]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'{phase.capitalize()} Confusion Matrix - Epoch {epoch}')
    plt.tight_layout()
    
    log_dict[f"{phase}/confusion_matrix"] = wandb.Image(fig)
    plt.close(fig)
    
    # False classifications
    log_dict[f"{phase}/false_classifications_count"] = len(metrics['false_classified'])
    
    if len(metrics['false_classified']) > 0:
        false_confidences = [fc['confidence'] for fc in metrics['false_classified']]
        log_dict.update({
            f"{phase}/false_classifications_avg_confidence": np.mean(false_confidences),
            f"{phase}/false_classifications_min_confidence": np.min(false_confidences),
            f"{phase}/false_classifications_max_confidence": np.max(false_confidences),
        })

    return log_dict

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    wandb.init(
        project="DeepLearning-StrategieDiscoverySupervisedEventBased",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "lr": LR,
            "patience": PATIENCE,
            "weight_decay": WEIGHT_DECAY
        }
    )

    # Load dataset
    data = np.load('aoe4_dataset.npz', allow_pickle=True)
    X_seq = data['X_seq']
    X_mask = data['X_mask']
    X_meta = data['X_meta']
    y = data['y']

    # Load vocabularies
    entity_vocab = data['entity_vocab'].item()
    event_vocab = data['event_vocab'].item()
    type_vocab = data['type_vocab'].item()
    age_vocab = data['age_vocab'].item()
    civ_vocab = data['civ_vocab'].item()
    enemy_civ_vocab = data['enemy_civ_vocab'].item()
    map_vocab = data['map_vocab'].item()
    label_vocab = data['label_vocab'].item()
    
    label_names = {v: k for k, v in label_vocab.items()}  # Reverse mapping

    # Train/val split
    X_seq_train, X_seq_val, X_mask_train, X_mask_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        X_seq, X_mask, X_meta, y, test_size=TEST_SIZE, random_state=42
    )

    idx_to_strat = {v: k for k, v in label_vocab.items()}
    unique_classes, class_counts = np.unique(y, return_counts=True)

    for class_id, count in zip(unique_classes, class_counts):
        class_name = idx_to_strat[class_id]
        percentage = (count / len(y)) * 100
        print(f"{class_name:<30} {class_id:<12} {count:<12} {percentage:>10.1f}%")

    if CLASS_WEIGHT_METHODE == 'inverse':
        weights  = 1 / class_counts
    elif CLASS_WEIGHT_METHODE == 'sqr':
        weights = 1.0 / class_counts ** 0.25 #np.sqrt(class_counts)
    weights = weights / weights.sum() * len(weights)
    print(f"Class weights: {weights}")
    class_weights = torch.FloatTensor(weights).to(device)

    print(f"\nClass weights for loss function:")
    print(f"{'Strategy Name':<30} {'Weight':<15}")
    print("=" * 45)
    for class_id in range(len(label_vocab)):
        class_name = idx_to_strat[class_id]
        print(f"{class_name:<30} {class_weights[class_id]:.4f}")
    
    # Create datasets and dataloaders
    train_dataset = AoE4Dataset(X_seq_train, X_mask_train, X_meta_train, y_train)
    val_dataset = AoE4Dataset(X_seq_val, X_mask_val, X_meta_val, y_val)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")


    # sampler = WeightedRandomSampler(class_weights , num_samples=len(class_weights ), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    vocab_sizes = {
        'entity': len(entity_vocab) + 1,
        'event': len(event_vocab) + 1,
        'type': len(type_vocab) +1,
        'age': len(age_vocab) + 1,
        'civ': len(civ_vocab) + 1,
        'enemy_civ': len(enemy_civ_vocab) + 1,
        'map': len(map_vocab) + 1
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StrategyGRU(vocab_sizes, num_classes=len(label_vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    # Training settings
    best_val_acc = 0
    patience_counter = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train
        train_acc, avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
        # Validation
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
            
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f' Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f' Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Log to wandb
        log_dict = log_metrics_to_wandb(val_metrics, label_names, epoch, phase='val')
        scheduler.step(val_loss)
 
        wandb.log(log_dict | {
            "Train Loss": avg_train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Epoch": epoch
        }, step=epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_strat_disc_event_based.pt')
            print(f'  → Model saved! (Val Acc: {val_acc:.2f}%)')
            # patience_counter = 0
        # else:
        #     patience_counter += 1
        #     print(f'  → No improvement ({patience_counter}/{PATIENCE})')
        
        # # Early stopping
        # if patience_counter >= PATIENCE:
        #     print(f'\nEarly stopping triggered after {epoch+1} epochs')
        #     break

    print(f'\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    
    torch.save(model.state_dict(), 'best_strat_disc_event_based.pt')
    wandb.save('best_strat_disc_event_based.pt')


if __name__ == "__main__":
    main()
