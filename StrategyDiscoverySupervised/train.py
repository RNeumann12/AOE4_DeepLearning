import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from StrategyDiscoverySupervised.dataset_resource_based import GameSequenceDatasetResourceBased
from StrategyDiscoverySupervised.dataset_event_based import GameSequenceDatasetEventBased
from model import StrategyGRU
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing for better regularization and class balance"""
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        """
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # KL divergence loss
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        # Apply class weights if provided
        if self.weight is not None:
            loss = loss * self.weight[targets]
        
        return loss.mean()

CSV_PATH = 'transformer_input_test_v2.csv'
SEQ_LEN = 100
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 7e-4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.5
LABEL_SMOOTHING=0.04
# CLASS_SCALE = 0.8
CLASS_WEIGHT_METHODE = 'sqr'
WEIGHT_DECAY = 0.01
DATASET = 'event_based'


def add_noise(tensor, noise_level=0.01):
    """Add Gaussian noise to tensor during training"""
    if tensor is None:
        return tensor
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_confidences = []

    pbar = tqdm(dataloader, desc="Epoch {epoch}Training", leave=False)
    for numeric_seq, event_seq, time_seq, mask, player_civ, enemy_civ, map_id, age, targets in pbar:
        numeric_seq = numeric_seq.to(device)
        event_seq = event_seq.to(device)
        time_seq = time_seq.to(device)
        mask = mask.to(device)
        player_civ = player_civ.to(device)
        enemy_civ = enemy_civ.to(device)
        map_id = map_id.to(device)
        age = age.to(device)
        targets = targets.to(device).long()

        numeric_seq = add_noise(numeric_seq, noise_level=0.01)
        time_seq = add_noise(time_seq, noise_level=0.01)
        optimizer.zero_grad()
        
        logits, _ = model(numeric_seq, event_seq, mask, time_seq, player_civ, enemy_civ, map_id, age)
        loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    for numeric_seq, event_seq, time_seq, mask, player_civ, enemy_civ, map_id, age, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        numeric_seq = numeric_seq.to(device)
        event_seq = event_seq.to(device)
        time_seq = time_seq.to(device)
        mask = mask.to(device)
        player_civ = player_civ.to(device)
        enemy_civ = enemy_civ.to(device)
        map_id = map_id.to(device)
        age = age.to(device)
        targets = targets.to(device).long()

        logits, _ = model(numeric_seq, event_seq, mask, time_seq, player_civ, enemy_civ, map_id, age)

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
    
    # table = wandb.Table(
    #     data=metrics_data,
    #     columns=[
    #         "Strategy", "TP", "FP", "FN", "TN",
    #         "TPR", "FPR", "FNR",
    #         "Precision", "Recall", "F1-Score", "Support"
    #     ]
    # )
    
    # wandb.log({f"{split_name}/Per-Class Detailed Metrics": table})
    
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
    
    # plt.tight_layout()
    # # wandb.log({f"{split_name}/Confidence Distributions": wandb.Image(fig)})
    # plt.close()
    
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
    
    # conf_table = wandb.Table(
    #     data=confidence_data,
    #     columns=["Strategy", "Avg Confidence", "Min Confidence", "Max Confidence", "Correct Avg Confidence", "Count"]
    # )
    # wandb.log({f"{split_name}/Per-Class Confidence Stats": conf_table})

def main():
    csv_path = CSV_PATH
    seq_len = SEQ_LEN 
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    hidden_size = HIDDEN_SIZE
    num_layers = NUM_LAYERS

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

    dataset = GameSequenceDatasetResourceBased(csv_path, seq_len=seq_len)
    
    # Extract targets from sequences for class distribution analysis
    dataset_targets = [seq['target'] for seq in dataset.sequences]
    
    # Debug: Check class distribution
    unique_classes, class_counts = np.unique(dataset_targets, return_counts=True)
    print(f"\nClass distribution in dataset:")
    print(f"{'Strategy Name':<30} {'Class ID':<12} {'Count':<12} {'Percentage':<12}")
    print("=" * 66)
    
    # Create reverse mapping from index to strategy name
    idx_to_strat = {v: k for k, v in dataset.strat_vocab.items()}
    
    for class_id, count in zip(unique_classes, class_counts):
        class_name = idx_to_strat[class_id]
        percentage = (count / len(dataset)) * 100
        print(f"{class_name:<30} {class_id:<12} {count:<12} {percentage:>10.1f}%")
    
    # Calculate class weights (inverse frequency) to handle imbalance
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
    for class_id in range(len(dataset.strat_vocab)):
        class_name = idx_to_strat[class_id]
        print(f"{class_name:<30} {class_weights[class_id]:.4f}")

    # Log class distribution and class weights to W&B (table + bar chart)
    # Prepare counts per class
    counts_by_class = [0] * len(dataset.strat_vocab)
    for cid, cnt in zip(unique_classes, class_counts):
        counts_by_class[cid] = int(cnt)

    # Build table data
    table_data = []
    for class_id in range(len(dataset.strat_vocab)):
        class_name = idx_to_strat[class_id]
        count = counts_by_class[class_id]
        percentage = (count / len(dataset)) * 100 if len(dataset) > 0 else 0.0
        weight = float(class_weights[class_id].item()) if hasattr(class_weights[class_id], 'item') else float(class_weights[class_id])
        table_data.append([class_name, class_id, count, f"{percentage:.2f}", weight])

    # table = wandb.Table(
    #     data=table_data,
    #     columns=["Strategy", "Class ID", "Count", "Percentage", "Weight"]
    # )
    # wandb.log({"Dataset/Class Distribution": table})

    # Bar chart of class counts
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [idx_to_strat[i] for i in range(len(dataset.strat_vocab))]
    ax.bar(labels, counts_by_class, color='tab:blue', edgecolor='black')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution (Counts)')
    plt.tight_layout()
    wandb.log({"Dataset/Class Distribution (Chart)": wandb.Image(fig)})
    plt.close()
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # # Build class count map
    # class_count_map = {cid: count for cid, count in zip(unique_classes, class_counts)}

    # # Compute log-damped per-sample weights: w = log(1 + N/count)
    # total_samples = len(dataset)
    # sample_weights_all = np.array([np.log(1.0 + total_samples / class_count_map[seq['target']]) for seq in dataset.sequences])

    # # Cap extreme weights using milder 5th/95th percentiles to avoid aggressive clipping
    # clip_low_pct = 5
    # clip_high_pct = 95
    # low_clip = float(np.percentile(sample_weights_all, clip_low_pct))
    # high_clip = float(np.percentile(sample_weights_all, clip_high_pct))
    # sample_weights_clipped = np.clip(sample_weights_all, low_clip, high_clip)

    # # Identify very rare classes and apply a small boost to their sample weights so they're not dropped entirely
    # rare_class_threshold = 0.10  # classes with <10% of samples are considered rare
    # rare_boost_factor = 1.25
    # rare_classes = [cid for cid, cnt in class_count_map.items() if (cnt / total_samples) < rare_class_threshold]

    # sample_weights_boosted = sample_weights_clipped.copy()
    # for rc in rare_classes:
    #     mask = np.array([1 if seq['target'] == rc else 0 for seq in dataset.sequences], dtype=bool)
    #     sample_weights_boosted[mask] = sample_weights_boosted[mask] * rare_boost_factor

    # # Log clipping info, boosted classes and distribution to W&B
    # wandb.log({
    #     "Sampler/weights_raw_min": float(sample_weights_all.min()),
    #     "Sampler/weights_raw_max": float(sample_weights_all.max()),
    #     "Sampler/weights_clip_low": low_clip,
    #     "Sampler/weights_clip_high": high_clip,
    #     "Sampler/rare_classes": str(rare_classes),
    #     "Sampler/rare_boost_factor": rare_boost_factor
    # })

    # fig_sw, ax_sw = plt.subplots(figsize=(8, 4))
    # ax_sw.hist(sample_weights_all, bins=50, alpha=0.5, label='raw', color='gray')
    # ax_sw.hist(sample_weights_boosted, bins=50, alpha=0.7, label='boosted', color='tab:blue')
    # ax_sw.legend()
    # ax_sw.set_title('Sampler Weights: Raw vs Boosted/Clipped')
    # plt.tight_layout()
    # wandb.log({"Sampler/Weights Distribution": wandb.Image(fig_sw)})
    # plt.close()

    # # For train subset, pick weights for its indices (use boosted weights)
    # train_indices = train_dataset.indices  # Subset
    # train_weights = [float(sample_weights_boosted[i]) for i in train_indices]

    sampler = WeightedRandomSampler(class_weights , num_samples=len(class_weights ), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = StrategyGRU(
        num_numeric=len(dataset.NUMERICAL_COLS),
        num_events=len(dataset.event_vocab),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(dataset.strat_vocab),
        dropout=DROPOUT,
        num_civs=len(dataset.civ_vocab),
        num_enemy_civs=len(dataset.enemy_civ_vocab),
        num_maps=len(dataset.map_vocab),
        num_ages=len(dataset.age_vocab) if dataset.age_vocab else 1
    ).to(device)

    # If using a WeightedRandomSampler, do NOT use class weights in the loss (sampler balances classes)
    # use_sampler = 'sampler' in locals()
    # if use_sampler:
    #     print("Using WeightedRandomSampler: disabling class weights in loss to avoid overcompensation.")
    #     # wandb.log({"Loss/Using Class Weights": False})
    #     criterion = LabelSmoothingLoss(num_classes=len(dataset.strat_vocab), smoothing=LABEL_SMOOTHING, weight=None)
    # else:
    #     # wandb.log({"Loss/Using Class Weights": True})
    #     criterion = LabelSmoothingLoss(num_classes=len(dataset.strat_vocab), smoothing=LABEL_SMOOTHING, weight=class_weights.to(device))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Early stopping
    best_val_acc = 0.0
    patience = 6
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, num_epochs+1):
        # Debug: Check model parameter changes
        param_before = next(model.parameters()).clone().detach()
        
        train_loss, train_acc, train_preds, train_targets, train_confidences = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Check if parameters changed
        param_after = next(model.parameters()).detach()
        param_changed = not torch.allclose(param_before, param_after)
        
        val_loss, val_acc, val_preds, val_targets, val_confidences = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)        
        wandb.log({
            "Train Loss (raw)": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss (raw)": val_loss,
            "Validation Accuracy": val_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Parameters updated: {param_changed}")
        
        # Debug: Check prediction distribution
        unique_preds, counts = np.unique(val_preds, return_counts=True)
        pred_dist = {f"class_{uid}": cnt for uid, cnt in zip(unique_preds, counts)}
        print(f"  Validation prediction distribution: {pred_dist}")
        print(f"  Validation unique predictions: {len(unique_preds)} out of {len(dataset.strat_vocab)} classes")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ Best validation accuracy: {best_val_acc:.4f}")
        # else:
        #     patience_counter += 1
        #     print(f"  ✗ No improvement. Patience: {patience_counter}/{patience}")
            
        #     if patience_counter >= patience:
        #         print(f"\n🛑 Early stopping triggered at epoch {epoch}")
        #         print(f"Best validation accuracy was: {best_val_acc:.4f}")
        #         model.load_state_dict(best_model_state)
        #         break
        
        # Log confusion matrices every epoch
        if True:
            plot_confusion_matrix(train_preds, train_targets, dataset, "Train")
            plot_confusion_matrix(val_preds, val_targets, dataset, "Validation")
        
        # Log classification metrics and confidence metrics every epoch
        if True:
            log_classification_metrics(train_preds, train_targets, dataset, "Train")
            log_classification_metrics(val_preds, val_targets, dataset, "Validation")
            log_confidence_metrics(train_preds, train_targets, train_confidences, dataset, "Train")
            log_confidence_metrics(val_preds, val_targets, val_confidences, dataset, "Validation")
    
    # Log final training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train/Val Loss curve
    axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(val_losses, label='Validation Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss difference (to show overfitting)
    train_losses_array = np.array(train_losses)
    val_losses_array = np.array(val_losses)
    loss_diff = val_losses_array - train_losses_array
    axes[0, 1].plot(loss_diff, label='Val Loss - Train Loss', marker='o', color='red')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Difference')
    axes[0, 1].set_title('Overfitting Indicator (positive = overfitting)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1, 0].plot(train_accs, label='Train Accuracy', marker='o')
    axes[1, 0].plot(val_accs, label='Validation Accuracy', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training and Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference (to show overfitting)
    loss_diff = val_losses_array - train_losses_array
    axes[1, 1].plot(loss_diff, label='Val Loss - Train Loss', marker='o', color='red')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].set_title('Overfitting Indicator (positive = overfitting)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"Training Curves": wandb.Image(fig)})
    plt.close()
    
    # Save model checkpoint
    torch.save(model.state_dict(), 'best_strategy_model.pt')
    wandb.save('best_strategy_model.pt')


if __name__ == "__main__":
    main()
