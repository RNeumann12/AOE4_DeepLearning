import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import GameSequenceDataset
from model import StrategyGRU
from tqdm import tqdm
import wandb

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
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({'Loss': total_loss / total, 'Acc': correct / total})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for numeric_seq, event_seq, mask, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        numeric_seq = numeric_seq.to(device)
        event_seq = event_seq.to(device)
        mask = mask.to(device)
        targets = targets.to(device).long()

        logits, _ = model(numeric_seq, event_seq, mask)

        loss = criterion(logits, targets)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    csv_path = CSV_PATH
    seq_len = SEQ_LEN 
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    hidden_size = HIDDEN_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

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

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=learning_rate), device)
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)

        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Epoch": epoch + 1
        })

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
