import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import GameSequenceDataset
from model import StrategyGRU
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = 'transformer_input_test_v2.csv'
SEQ_LEN = 50
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 128

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_confidences = []

    pbar = tqdm(dataloader, desc="Epoch {epoch}Training", leave=False)
    for numeric_seq, event_seq, mask, targets in pbar:
        numeric_seq = numeric_seq.to(device)
        event_seq = event_seq.to(device)
        mask = mask.to(device)
        targets = targets.to(device).long()

        # print(numeric_seq.shape)
        # print(event_seq.shape)
        # print(mask.shape)
        # print(targets.shape, targets.dtype)

        optimizer.zero_grad()
        
        logits, _ = model(numeric_seq, event_seq, mask)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        confidences = torch.nn.functional.softmax(logits, dim=1).max(dim=1)[0]
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_confidences.extend(confidences.cpu().detach().numpy())

        pbar.set_postfix({'Loss': total_loss / total, 'Acc': correct / total})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets, all_confidences

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_confidences = []

    for numeric_seq, event_seq, mask, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        numeric_seq = numeric_seq.to(device)
        event_seq = event_seq.to(device)
        mask = mask.to(device)
        targets = targets.to(device).long()

        logits, _ = model(numeric_seq, event_seq, mask)

        loss = criterion(logits, targets)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        confidences = torch.nn.functional.softmax(logits, dim=1).max(dim=1)[0]
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_confidences.extend(confidences.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets, all_confidences

def plot_confusion_matrix(preds, targets, dataset, split_name):
    """Generate and log confusion matrix to W&B"""
    cm = confusion_matrix(targets, preds)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=dataset.strat_vocab.values(),
        yticklabels=dataset.strat_vocab.values(),
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    plt.title(f'Confusion Matrix - {split_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Log to W&B
    wandb.log({f"{split_name}/Confusion Matrix": wandb.Image(fig)})
    plt.close()

def log_classification_metrics(preds, targets, dataset, split_name):
    """Log detailed classification metrics to W&B"""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, average=None, labels=list(range(len(dataset.strat_vocab))), zero_division=0
    )
    
    # Calculate per-class confusion matrix values
    cm = confusion_matrix(targets, preds, labels=list(range(len(dataset.strat_vocab))))
    
    # Create reverse mapping from index to class name
    idx_to_strat = {v: k for k, v in dataset.strat_vocab.items()}
    
    # Create per-class metrics table with TP, FP, FN and rates
    metrics_data = []
    for class_idx in range(len(dataset.strat_vocab)):
        class_name = idx_to_strat[class_idx]
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        metrics_data.append([
            class_name,
            int(tp),
            int(fp),
            int(fn),
            int(tn),
            f"{tpr:.4f}",
            f"{fpr:.4f}",
            f"{fnr:.4f}",
            f"{precision[class_idx]:.4f}",
            f"{recall[class_idx]:.4f}",
            f"{f1[class_idx]:.4f}",
            int(support[class_idx])
        ])
    
    table = wandb.Table(
        data=metrics_data,
        columns=[
            "Strategy", "TP", "FP", "FN", "TN",
            "TPR", "FPR", "FNR",
            "Precision", "Recall", "F1-Score", "Support"
        ]
    )
    
    wandb.log({f"{split_name}/Per-Class Detailed Metrics": table})
    
    # Log macro and weighted averages
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    wandb.log({
        f"{split_name}/Macro F1-Score": macro_f1,
        f"{split_name}/Weighted F1-Score": weighted_f1,
    })

def log_confidence_metrics(preds, targets, confidences, dataset, split_name):
    """Log confidence-based metrics to W&B"""
    confidences = np.array(confidences)
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Overall confidence statistics
    correct_mask = preds == targets
    incorrect_mask = preds != targets
    
    correct_confidences = confidences[correct_mask] if correct_mask.sum() > 0 else []
    incorrect_confidences = confidences[incorrect_mask] if incorrect_mask.sum() > 0 else []
    
    # Log confidence metrics
    wandb.log({
        f"{split_name}/Overall Avg Confidence": np.mean(confidences),
        f"{split_name}/Overall Min Confidence": np.min(confidences),
        f"{split_name}/Overall Max Confidence": np.max(confidences),
        f"{split_name}/Correct Predictions Avg Confidence": np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
        f"{split_name}/Incorrect Predictions Avg Confidence": np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0,
    })
    
    # Plot confidence distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of all confidences
    axes[0].hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{split_name} - Confidence Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Comparison of correct vs incorrect
    if len(correct_confidences) > 0:
        axes[1].hist(correct_confidences, bins=50, alpha=0.6, label='Correct', edgecolor='black')
    if len(incorrect_confidences) > 0:
        axes[1].hist(incorrect_confidences, bins=50, alpha=0.6, label='Incorrect', edgecolor='black')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{split_name} - Correct vs Incorrect Predictions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({f"{split_name}/Confidence Distributions": wandb.Image(fig)})
    plt.close()
    
    # Per-class confidence statistics
    idx_to_strat = {v: k for k, v in dataset.strat_vocab.items()}
    confidence_data = []
    for class_idx in range(len(dataset.strat_vocab)):
        class_name = idx_to_strat[class_idx]
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_confidences = confidences[class_mask]
            class_correct_mask = (targets == preds) & class_mask
            
            avg_conf = np.mean(class_confidences)
            min_conf = np.min(class_confidences)
            max_conf = np.max(class_confidences)
            correct_conf = np.mean(confidences[class_correct_mask]) if class_correct_mask.sum() > 0 else 0
            
            confidence_data.append([
                class_name,
                f"{avg_conf:.4f}",
                f"{min_conf:.4f}",
                f"{max_conf:.4f}",
                f"{correct_conf:.4f}",
                int(class_mask.sum())
            ])
    
    conf_table = wandb.Table(
        data=confidence_data,
        columns=["Strategy", "Avg Confidence", "Min Confidence", "Max Confidence", "Correct Avg Confidence", "Count"]
    )
    wandb.log({f"{split_name}/Per-Class Confidence Stats": conf_table})

def main():
    csv_path = CSV_PATH
    seq_len = SEQ_LEN 
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    hidden_size = HIDDEN_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # Clear any cached memory before starting
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    wandb.init(
        project="DeepLearning-StrategieDiscoverySupervised",
        config={
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "lr": learning_rate,
            "hidden_size": hidden_size
        }
    )

    dataset = GameSequenceDataset(csv_path, seq_len=seq_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = StrategyGRU(
        num_numeric=len(dataset.NUMERICAL_COLS),
        num_events=len(dataset.event_vocab),
        hidden_size=hidden_size,
        num_layers=2,
        num_classes=len(dataset.strat_vocab),
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc, train_preds, train_targets, train_confidences = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets, val_confidences = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Log basic metrics
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log confusion matrices every 5 epochs
        if epoch % 5 == 0:
            plot_confusion_matrix(train_preds, train_targets, dataset, "Train")
            plot_confusion_matrix(val_preds, val_targets, dataset, "Validation")
        
        # Log classification metrics and confidence metrics every 5 epochs
        if epoch % 5 == 0:
            log_classification_metrics(train_preds, train_targets, dataset, "Train")
            log_classification_metrics(val_preds, val_targets, dataset, "Validation")
            log_confidence_metrics(train_preds, train_targets, train_confidences, dataset, "Train")
            log_confidence_metrics(val_preds, val_targets, val_confidences, dataset, "Validation")
    
    # Log final training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"Training Curves": wandb.Image(fig)})
    plt.close()
    
    # Save model checkpoint
    torch.save(model.state_dict(), 'best_strategy_model.pth')
    wandb.save('best_strategy_model.pth')


if __name__ == "__main__":
    main()
