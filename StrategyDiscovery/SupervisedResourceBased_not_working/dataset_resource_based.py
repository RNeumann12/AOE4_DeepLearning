import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class GameSequenceDatasetResourceBased(Dataset):
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
    
    # Event columns - all finished_* events
    EVENT_COLS = [
        'finished_buildings',
        'finished_units',
        'finished_upgrades',
#        'finished_animals'
    ]
    
    # Numerical/resource columns
    NUMERICAL_COLS = [
        #'food', 'wood', 'gold', 'stone', 
        # 'military', 'economy', 'technology', 'society',
        # 'food_per_min', 'wood_per_min', 'gold_per_min', 'stone_per_min',
        'villager_delta', 
        #'oliveoil', 'oliveoil_per_min'
    ]
    
    def __init__(self, csv_path, seq_len=50, target_col='strat'):
        """
        Args:
            csv_path (str): path to CSV containing game sequences
            seq_len (int): max sequence length (pad/truncate sequences)
            target_col (str): column name for target label (strategy)
        """
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.target_col = target_col
        
        # Identify event columns (those starting with 'finished_')
        self.event_cols = [col for col in self.df.columns if col.startswith('finished_')]
        
        # Check for time/age columns
        self.has_time = 'time' in self.df.columns
        self.has_age = 'age' in self.df.columns
        
        print(f"Loaded CSV with {len(self.df)} rows")
        print(f"Found {len(self.event_cols)} event columns: {self.event_cols}")
        print(f"Found {len(self.NUMERICAL_COLS)} numerical columns")
        print(f"Has 'time' column: {self.has_time}")
        print(f"Has 'age' column: {self.has_age}")
        
        # Build vocabularies
        self.event_vocab = self._build_event_vocab()
        self.strat_vocab = {s: i for i, s in enumerate(sorted(self.df[target_col].unique()))}
        self.civ_vocab = {c: i for i, c in enumerate(sorted(self.df['player_civ'].unique()))}
        self.enemy_civ_vocab = {c: i for i, c in enumerate(sorted(self.df['enemy_civ'].unique()))}
        self.map_vocab = {m: i for i, m in enumerate(sorted(self.df['map'].unique()))}
        
        # Age vocabulary (if exists)
        if self.has_age:
            self.age_vocab = {a: i for i, a in enumerate(sorted(self.df['age'].dropna().unique()))}
        else:
            self.age_vocab = {}
        
        print(f"\nVocabulary sizes:")
        print(f"  Events: {len(self.event_vocab)}")
        print(f"  Strategies: {len(self.strat_vocab)}")
        print(f"  Civilizations: {len(self.civ_vocab)}")
        print(f"  Enemy Civs: {len(self.enemy_civ_vocab)}")
        print(f"  Maps: {len(self.map_vocab)}")
        if self.has_age:
            print(f"  Ages: {len(self.age_vocab)} - {list(self.age_vocab.keys())}")
        
        # Group by player-game and build sequences
        self.plays = self.df.groupby(['game_id', 'profile_id'])
        self.sequences = self._build_sequences()
        
        print(f"\nBuilt {len(self.sequences)} sequences")
    
    def _build_event_vocab(self):
        """Build vocabulary from all events in all event columns"""
        vocab = {'<PAD>': 0}  # Reserved for padding
        idx = 1
        
        for col in self.event_cols:
            # Process each event column
            for events_str in self.df[col].dropna():
                if pd.isna(events_str):
                    continue
                
                # Split semicolon-separated events
                events = [e.strip() for e in str(events_str).split(';') if e.strip()]
                for event in events:
                    if event not in vocab:
                        vocab[event] = idx
                        idx += 1
        
        print(f"Built event vocab with {len(vocab)} unique events")
        return vocab
    
    def _encode_events(self, events_str):
        """Convert semicolon-separated event string to list of vocab indices"""
        if pd.isna(events_str) or events_str == '':
            return []
        
        events = [e.strip() for e in str(events_str).split(';') if e.strip()]
        encoded = []
        for event in events:
            if event in self.event_vocab:
                encoded.append(self.event_vocab[event])
            else:
                # Unknown event - use PAD token
                encoded.append(0)
        
        return encoded
    
    def _encode_numeric(self, value):
        """Safely encode numeric value, handling NaN"""
        if pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _build_sequences(self):
        """Build sequences from grouped games"""
        sequences = []
        
        for (game_id, profile_id), group in tqdm(self.plays, total=len(self.plays), desc="Building sequences"):
            # Sort by time to maintain temporal order
            group = group.sort_values('time').reset_index(drop=True)
            
            num_steps = min(len(group), self.seq_len)
            
            # Initialize tensors
            # Numeric features: (seq_len, num_resources)
            seq_numeric = torch.zeros(self.seq_len, len(self.NUMERICAL_COLS), dtype=torch.float32)
            
            # Event features: (seq_len, event_vocab_size) - multi-hot encoding
            seq_events = torch.zeros(self.seq_len, len(self.event_vocab), dtype=torch.float32)
            
            # Mask: (seq_len) - indicates which timesteps are valid
            seq_mask = torch.zeros(self.seq_len, dtype=torch.bool)
            
            # Age/time encoding: (seq_len) - normalized time in game
            seq_time = torch.zeros(self.seq_len, dtype=torch.float32)
            
            # Extract scalar features (same for all timesteps in this game)
            player_civ = group['player_civ'].iloc[0]
            enemy_civ = group['enemy_civ'].iloc[0]
            map_id = group['map'].iloc[0]
            target_strat = group[self.target_col].iloc[0]
            age = group['age'].iloc[0] if self.has_age else None
            
            # Encode scalar features
            player_civ_idx = self.civ_vocab.get(player_civ, 0)
            enemy_civ_idx = self.enemy_civ_vocab.get(enemy_civ, 0)
            map_idx = self.map_vocab.get(map_id, 0)
            age_idx = self.age_vocab.get(age, 0) if self.has_age else 0
            target_idx = self.strat_vocab.get(target_strat, 0)
            
            # Get max time for normalization
            max_time = 1.0
            if self.has_time:
                max_time = group['time'].max() if 'time' in group.columns else 1.0
                max_time = max(max_time, 1.0)  # Avoid division by zero
            
            # Fill in sequences for each timestep
            for i, (_, row) in enumerate(group.iterrows()):
                if i >= self.seq_len:
                    break
                
                # 1. Numeric features (resources, economy, etc.)
                numeric_values = [self._encode_numeric(row[col]) for col in self.NUMERICAL_COLS]
                seq_numeric[i] = torch.tensor(numeric_values, dtype=torch.float32)
                
                # 2. Event features (count-based encoding - handles multiple occurrences)
                event_vec = torch.zeros(len(self.event_vocab), dtype=torch.float32)
                for col in self.event_cols:
                    event_indices = self._encode_events(row[col])
                    # Count occurrences of each event (e.g., Sheep appearing twice = 2.0)
                    for event_idx in event_indices:
                        if event_idx > 0:  # Skip padding
                            event_vec[event_idx] += 1.0
                seq_events[i] = event_vec
                
                # 3. Time/Age encoding (normalized)
                if self.has_time and 'time' in row.index:
                    time_val = self._encode_numeric(row['time'])
                    seq_time[i] = time_val / max_time
                elif self.has_age and 'age' in row.index:
                    age_val = self._encode_numeric(row['age'])
                    seq_time[i] = age_val / max_time
                else:
                    # Use index as proxy for time if neither exists
                    seq_time[i] = i / self.seq_len
                
                # 4. Valid timestep mask
                seq_mask[i] = True
            
            # Store sequence
            sequence = {
                'numeric': seq_numeric,           # (seq_len, num_resources)
                'events': seq_events,             # (seq_len, event_vocab_size)
                'time': seq_time,                 # (seq_len,)
                'mask': seq_mask,                 # (seq_len,)
                'player_civ': player_civ_idx,     # scalar
                'enemy_civ': enemy_civ_idx,       # scalar
                'map': map_idx,                   # scalar
                'age': age_idx,                   # scalar
                'target': target_idx,             # scalar (strategy)
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (numeric, events, time, mask, player_civ, enemy_civ, map, age, target)
                - numeric: (seq_len, num_resources)
                - events: (seq_len, event_vocab_size) - count-based encoding (Sheep appearing twice = 2.0)
                - time: (seq_len,)
                - mask: (seq_len,)
                - player_civ: scalar (int)
                - enemy_civ: scalar (int)
                - map: scalar (int)
                - age: scalar (int) - categorical age/epoch
                - target: scalar (int) - strategy label
        """
        seq = self.sequences[idx]
        
        return (
            seq['numeric'],
            seq['events'],
            seq['time'],
            seq['mask'],
            torch.tensor(seq['player_civ'], dtype=torch.long),
            torch.tensor(seq['enemy_civ'], dtype=torch.long),
            torch.tensor(seq['map'], dtype=torch.long),
            torch.tensor(seq['age'], dtype=torch.long),
            torch.tensor(seq['target'], dtype=torch.long)
        )


if __name__ == "__main__":
    # Test the dataset
    dataset = GameSequenceDatasetResourceBased('/TrainingData/input_resource_based.csv', seq_len=50)
    print(f"\n✓ Dataset created successfully!")
    print(f"Total samples: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    numeric, events, time, mask, player_civ, enemy_civ, map_id, target = sample
    
    print(f"\nSample 0:")
    print(f"  Numeric shape: {numeric.shape} (resources)")
    print(f"  Events shape: {events.shape} (multi-hot events)")
    print(f"  Time shape: {time.shape} (game phase)")
    print(f"  Mask shape: {mask.shape} (valid timesteps)")
    print(f"  Player civ: {player_civ.item()} (vocab size: {len(dataset.civ_vocab)})")
    print(f"  Enemy civ: {enemy_civ.item()} (vocab size: {len(dataset.enemy_civ_vocab)})")
    print(f"  Map: {map_id.item()} (vocab size: {len(dataset.map_vocab)})")
    print(f"  Target strategy: {target.item()} (vocab size: {len(dataset.strat_vocab)})")
    
    # Check non-zero events
    events_per_step = events.sum(dim=1)
    print(f"\nEvents per timestep (should not be all 0):")
    print(f"  Min: {events_per_step.min().item():.0f}")
    print(f"  Max: {events_per_step.max().item():.0f}")
    print(f"  Mean: {events_per_step[events_per_step > 0].mean().item():.2f} (where > 0)")
    print(f"  Timesteps with events: {(events_per_step > 0).sum().item()}/{len(events)}")
