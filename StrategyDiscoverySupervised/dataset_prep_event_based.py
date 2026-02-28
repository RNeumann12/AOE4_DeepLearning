import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

# Event columns - all finished_* events
VOCAB_COLS = [
    'player_civ',
    'enemy_civ',
    'map',
    'player_result',
    'event',
    'entity',
    'type',
    'age',
]

TARGET_COL = 'strat'


# Numerical/resource columns
NUMERICAL_COLS = [
    'villager', 
]


MAX_LEN = 150
PAD_DATA = True
MASK_DATA = True

class DataSetPrep(Dataset):
    """
    Enhanced PyTorch Dataset for AoE4 player sequences.
    
    Includes:
    - Civilizations (player_civ, enemy_civ)
    - Map information
    - Resources (food, wood, gold, stone, oliveoil)
    - Events (finished_buildings, finished_units, finished_upgrades, finished_animals)
    - Age/Time information
    - Target strategy (strat)
    """
    
    def __init__(self, csv_path):
        """
        Args:
            csv_path (str): path to CSV containing game sequences
            seq_len (int): max sequence length (pad/truncate sequences)
            target_col (str): column name for target label (strategy)
        """
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")


        grouped = df.groupby(['game_id', 'profile_id'])
        print(f"Grouped {len(grouped)} rows")

        player_result_vocab = {val: idx + 1 for idx, val in enumerate(df['player_result'].unique())}
        entity_vocab = {val: idx + 1 for idx, val in enumerate(df['entity'].unique())}
        event_vocab = {val: idx + 1 for idx, val in enumerate(df['event'].unique())}
        type_vocab = {val: idx + 1 for idx, val in enumerate(df['type'].unique())}
        age_vocab = {val: idx + 1 for idx, val in enumerate(df['age'].unique())}
        civ_vocab = {val: idx + 1 for idx, val in enumerate(df['player_civ'].unique())}
        enemy_civ_vocab = {val: idx + 1 for idx, val in enumerate(df['enemy_civ'].unique())}
        map_vocab = {val: idx + 1 for idx, val in enumerate(df['map'].unique())}
        label_vocab = {val: idx for idx, val in enumerate(df['strat'].unique())} 
        
        
        print(f"\nVocabulary sizes:")
        print(f"  Entity: {len(entity_vocab)}")
        print(f"  Events: {len(event_vocab)}")
        print(f"  Type: {len(type_vocab)}")
        print(f"  Ages: {len(age_vocab)} - {list(age_vocab.keys())}")

        print(f"  Civilizations: {len(civ_vocab)}")
        print(f"  Enemy Civs: {len(enemy_civ_vocab)}")
        print(f"  Maps: {len(map_vocab)}")
        print(f"  Strategies: {len(label_vocab)}  - {list(label_vocab.keys())}")
    
        sequences = []
        labels = []
        masks = []
        metadata = []
        game_ids = []
        player_ids = []

        # build sequences for event data, build metadata and label
        for (game_id, profile_id), group in grouped:
            group = group.sort_values('time')
            
            # Sequence features (per event)
            entity_seq = [entity_vocab[e] for e in group['entity']]
            event_seq = [event_vocab[e] for e in group['event']]
            type_seq = [type_vocab[e] for e in group['type']]
            age_seq = [age_vocab[a] for a in group['age']]
            time_seq = group['time'].values
            villager_seq = group['villagers'].values
            
            # Stack: [entity_id, event_id, age_id, time, villagers]
            seq_features = np.column_stack([entity_seq, event_seq, type_seq, age_seq, time_seq, villager_seq])
            sequences.append(seq_features)
            
            mask = np.ones(len(seq_features))
            masks.append(mask)

            # Game-level metadata (static per game)
            meta = {
                'player_civ': civ_vocab[group.iloc[0]['player_civ']],
                'enemy_civ': enemy_civ_vocab[group.iloc[0]['enemy_civ']],
                'map': map_vocab[group.iloc[0]['map']]
            }
            metadata.append(meta)
            game_ids.append(game_id)
            player_ids.append(profile_id)
            
            # Label
            labels.append(label_vocab[group.iloc[0]['strat']])
 
        # Pad sequences and masks
        max_len = max(len(s) for s in sequences)
        if PAD_DATA == True:
            X_seq = np.array([np.pad(s, ((0, max_len - len(s)), (0, 0)), constant_values=0) 
                        for s in sequences])
        else:
            X_seq = sequences

        if MASK_DATA == True:
            X_mask = np.array([np.pad(m, (0, max_len - len(m)), constant_values=0) 
                        for m in masks])
        else:
            X_mask = None
            
        X_meta = np.array([[m['player_civ'], m['enemy_civ'], m['map']] for m in metadata])
        y = np.array(labels)

        self.X_seq = torch.FloatTensor(X_seq)
        self.X_mask = torch.FloatTensor(X_mask)
        self.X_meta = torch.LongTensor(X_meta)
        self.y = torch.LongTensor(y)

        # Save processed dataset
        np.savez('aoe4_dataset.npz', 
                X_seq=X_seq, 
                X_mask=X_mask, 
                X_meta=X_meta, 
                y=y,
                game_ids=np.array(game_ids),
                player_ids=np.array(player_ids),
                entity_vocab=entity_vocab,
                event_vocab=event_vocab,
                type_vocab=type_vocab,
                age_vocab=age_vocab,
                civ_vocab=civ_vocab,
                enemy_civ_vocab=enemy_civ_vocab,
                map_vocab=map_vocab,
                player_result_vocab=player_result_vocab,
                label_vocab=label_vocab)

        print(f"Dataset created! Shape: X_seq={self.X_seq.shape}, X_mask={self.X_mask.shape}, X_meta={self.X_meta.shape}, y={self.y.shape}")
