import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm 

class GameSequenceDataset(Dataset):
    EVENT_COLS = [
        'finished_buildings',
        'finished_units',
        'finished_upgrades',
        'finished_animals'
    ]

    NUMERICAL_COLS = [
        'food', 'wood', 'gold', 'stone', 'military', 'economy', 'technology', 'society',
        'food_per_min', 'wood_per_min', 'gold_per_min', 'stone_per_min',
        'villager_delta', 'oliveoil', 'oliveoil_per_min'
    ]
   
    """
    PyTorch Dataset for AoE4 player sequences.

    Each sample corresponds to one player in one game,
    with sequences of events sampled every 20 seconds.
    """
    def __init__(self, csv_path, seq_len = 50, target_col='strat'):
        """
        Args:
            csv_file (str): path to CSV containing game sequences
            seq_len (int): max sequence length (pad/truncate sequences)
            event_cols (list): columns containing semicolon-separated events (buildings, units, upgrades, animals)
        """

        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.event_cols = [col for col in self.df.columns if col.startswith('finished_')]
        self.target_col = target_col

        # Build vocabulary for categorical events
        self.event_vocab = self.build_event_vocab()
        self.strat_vocab = {s: i for i, s in enumerate(self.df[target_col].unique())}
        self.civ_vocab = {c: i for i, c in enumerate(self.df['player_civ'].unique())}
        self.enemy_civ_vocab = {c: i for i, c in enumerate(self.df['enemy_civ'].unique())}
        self.map_vocab = {m: i for i, m in enumerate(self.df['map'].unique())}
        self.vocab_size = len(self.event_vocab)

        # group by player-game for sequences
        self.plays = self.df.groupby(['game_id','profile_id'])

        self.sequences, self.masks, self.targets = self.build_sequences()


    def build_event_vocab(self):
        vocab = {'<PAD>': 0}  # zero reserved for padding
        idx = 1
        for col in self.event_cols:
            for events in self.df[col].dropna():
                for e in events.split(';'):
                    e = e.strip()
                    if e and e not in vocab:
                        vocab[e] = idx
                        idx += 1
        return vocab

    def encode_events(self, events_str):
        """Convert semicolon-separated events to indices"""
        if pd.isna(events_str):
            return []
        return [self.event_vocab[e.strip()] for e in events_str.split(';') if e.strip()]
    
    def build_sequences(self):
        sequences = []
        masks = []
        targets = []
        numeric_cols = self.NUMERICAL_COLS

        for (game_id, profile_id), group in tqdm(self.plays, total=len(self.plays), desc="Building sequences"):
            # Sort by timestamp
            group = group.sort_values('time')

            # basically zeros rightnow
            seq_numeric = torch.zeros(self.seq_len, len(numeric_cols), dtype=torch.float)
            seq_events = torch.zeros(self.seq_len, len(self.event_vocab), dtype=torch.float)
            mask = torch.zeros(self.seq_len, dtype=torch.bool)

            for i, (_, row) in enumerate(group.iterrows()):
                if i >= self.seq_len:
                    break
                # Numeric features
                seq_numeric[i] = torch.tensor(row.loc[numeric_cols].astype(float).values, dtype=torch.float)
                
                # Event features (multi-hot)
                event_vec = torch.zeros(len(self.event_vocab))
                for col in self.event_cols:
                    for idx in self.encode_events(row[col]):
                        event_vec[idx] = 1.0
                seq_events[i] = event_vec

                mask[i] = 1

            sequences.append({'numeric': seq_numeric, 'events': seq_events})
            masks.append(mask)
            targets.append(self.strat_vocab[row[self.target_col]])

        return sequences, masks, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        mask = self.masks[idx]
        target = self.targets[idx]
        return seq['numeric'], seq['events'], mask, target



# if __name__ == "__main__":
#     dataset = GameSequenceDataset('transformer_input_test_v2.csv', seq_len=50)
#     print(f"Dataset size: {len(dataset)} samples")
#     print(f"Vocabulary size: {dataset.vocab_size} unique events")
#     sample_numeric, sample_events, sample_mask, sample_y = dataset[0]
#     print(f"Sample numeric X shape: {sample_numeric.shape}")
#     print(f"Sample events X shape: {sample_events.shape}")
#     print(f"Sample mask shape: {sample_mask.shape}")
#     print(f"Sample y: {sample_y}")
