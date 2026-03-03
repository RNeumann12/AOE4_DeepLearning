import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_event_based import AoE4Dataset
from model_v2 import StrategyGRU
import random

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MODEL_PATH = 'StrategyDiscovery/SupervisedEventBased/strat_disc_event_based_final_model.pt'
DATA_PATH = 'StrategyDiscovery/SupervisedEventBased/aoe4_dataset_infer.npz'
VOCAB_PATH = 'StrategyDiscovery/SupervisedEventBased/aoe4_dataset_final.npz'

# --- LOAD DATA ---
data = np.load(DATA_PATH, allow_pickle=True)
vocab_data = np.load(VOCAB_PATH, allow_pickle=True)
X_seq = data['X_seq']
X_mask = data['X_mask']
X_meta = data['X_meta']
y = data['y']

# Load vocabularies
entity_vocab = vocab_data['entity_vocab'].item()
event_vocab = vocab_data['event_vocab'].item()
type_vocab = vocab_data['type_vocab'].item()
age_vocab = vocab_data['age_vocab'].item()
civ_vocab = vocab_data['civ_vocab'].item()
enemy_civ_vocab = vocab_data['enemy_civ_vocab'].item()
map_vocab = vocab_data['map_vocab'].item()
label_vocab = vocab_data['label_vocab'].item()

idx = random.randint(0, len(y) - 1)
sequence = torch.tensor(X_seq[idx]).unsqueeze(0).to(DEVICE)   # add batch dim
mask = torch.tensor(X_mask[idx]).unsqueeze(0).to(DEVICE)
metadata = torch.tensor(X_meta[idx]).unsqueeze(0).to(DEVICE)
# true_label = idx_to_label[y[idx]]

idx_to_label = {v: k for k, v in label_vocab.items()}

# --- CREATE DATASET AND DATALOADER ---
dataset = AoE4Dataset(X_seq, X_mask, X_meta, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- INITIALIZE MODEL ---
vocab_sizes = {
    'entity': len(entity_vocab) + 1,
    'event': len(event_vocab) + 1,
    'type': len(type_vocab) + 1,
    'age': len(age_vocab) + 1,
    'civ': len(civ_vocab) + 1,
    'enemy_civ': len(enemy_civ_vocab) + 1,
    'map': len(map_vocab) + 1
}

model = StrategyGRU(vocab_sizes, num_classes=len(label_vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    outputs = model(sequence, mask, metadata)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred_idx = probs.max(1)
    predicted_label = idx_to_label[pred_idx.item()]

game_id = data['game_ids'][idx]
player_id = data['player_ids'][idx]

print(f"Game ID:         {game_id}")
print(f"Player ID:       {player_id}")
print(f"Predicted label: {predicted_label}")
print(f"Confidence:      {confidence.item():.4f}")
print(f"Probabilities:   {probs.cpu().numpy()[0]}")
