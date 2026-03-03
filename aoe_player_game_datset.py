"""
Preprocessing utilities and PyTorch Dataset for AoE4 win prediction.

- Groups events by (game_id, profile_id)
- Builds vocabularies for "entity" tokens and "event" tokens
- Encodes sequences into token ids plus numeric/time features
- Exposes a Dataset that yields: {tokens, times, civ_id, enemy_civ_id, continuous_features, label}

Usage:
  from data_transformer import AoEEventDataset, build_vocabs, collate_fn

"""
import csv
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Sequence
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocabs(df: pd.DataFrame, min_freq: int = 1, 
                 filter_events: Optional[List[str]] = None,
                 filter_entities: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
   
    """
    Builds three vocabularies for "entity", "event" and "civ" columns in the given DataFrame.

    The vocabularies are built by counting the frequency of each token in the columns.
    Tokens with a frequency below the given min_freq are not included in the vocabularies.
    The vocabularies are returned as a dictionary with keys 'entity_vocab', 'event_vocab', 'civ_vocab', and 'map_vocab'.
    Each vocabulary is a dictionary mapping tokens to their indices.

    :param df: A DataFrame containing the columns 'entity', 'event', 'player_civ', and 'enemy_civ'.
    :param min_freq: The minimum frequency of a token to be included in the vocabularies.
    :param filter_events: List of event types to EXCLUDE (e.g., ['DESTROY']).
    :param filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep']).
    :return: A dictionary containing the built vocabularies.
    """
    # Apply filtering before building vocabularies
    original_len = len(df)
    
    # Filter out unwanted events (e.g., DESTROY)
    if filter_events:
        df = df[~df['event'].isin(filter_events)]
        filtered_len = len(df)
        print(f"Vocabularies: Filtered out events {filter_events}: {original_len - filtered_len} rows excluded")
    
    # Filter out unwanted entities (e.g., Sheep - captured, not built)
    if filter_entities:
        pre_filter_len = len(df)
        df = df[~df['entity'].astype(str).isin(filter_entities)]
        filtered_len = len(df)
        print(f"Vocabularies: Filtered out entities {filter_entities}: {pre_filter_len - filtered_len} rows excluded")
    
    # Filter out events at timestamp 0 (outside player control - starting units, etc.)
    if 'time' in df.columns:
        pre_filter_len = len(df)
        df = df[df['time'] != 0]
        filtered_len = len(df)
        if pre_filter_len - filtered_len > 0:
            print(f"Vocabularies: Filtered out timestamp=0 events: {pre_filter_len - filtered_len} rows excluded")
    
    entity_counts = Counter(df['entity'].astype(str).tolist())
    event_counts = Counter(df['event'].astype(str).tolist())
    civ_counts = Counter(df['player_civ'].astype(str).tolist() + df['enemy_civ'].astype(str).tolist())
    # map column may be absent in older/alternate CSVs; handle gracefully
    if 'map' in df.columns:
        map_counts = Counter(df['map'].astype(str).fillna(UNK_TOKEN).tolist())
    else:
        map_counts = Counter()

    def make_vocab(counter: Counter) -> Dict[str, int]:
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for tok, cnt in counter.items():
            if cnt >= min_freq:
                vocab[tok] = len(vocab)
        return vocab

    return {
        'entity_vocab': make_vocab(entity_counts),
        'event_vocab': make_vocab(event_counts),
        'civ_vocab': make_vocab(civ_counts),
        'map_vocab': make_vocab(map_counts),
    }


def encode_token(token: str, vocab: Dict[str, int]) -> int:
    return vocab.get(token, vocab.get(UNK_TOKEN, 1))

class AoEEventDataset(Dataset):
    """Dataset of sequences per player-game.

    Each item corresponds to a single player's event sequence in a game, with label in `player_won`.
    """

    def __init__(self, csv_path: str, entity_vocab: Dict[str, int], event_vocab: Dict[str, int], civ_vocab: Dict[str, int], map_vocab: Dict[str, int], max_len: Optional[int] = None, use_time_features: bool = True, truncation_strategy: str = 'head_tail',
                 filter_events: Optional[List[str]] = None, filter_entities: Optional[List[str]] = None):
        """
        Initialize the dataset from a CSV file containing event sequences per player-game.

        :param csv_path: Path to the CSV file containing the event sequences.
        :param entity_vocab: Vocabulary for "entity" tokens.
        :param event_vocab: Vocabulary for "event" tokens.
        :param civ_vocab: Vocabulary for "civ" tokens.
        :param map_vocab: Vocabulary for "map" tokens.
        :param max_len: Maximum length of a sequence. If None, do not truncate and use all events.
                        If set, sequences longer than `max_len` will be truncated according to
                        `truncation_strategy`.
        :param use_time_features: If True, use delta time scaled as a feature.
        :param truncation_strategy: One of {'head', 'tail', 'head_tail'} determining how to
                                    truncate sequences when they exceed `max_len`.
        :param filter_events: List of event types to EXCLUDE (e.g., ['DESTROY']).
        :param filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep']).
        """
        self.df = pd.read_csv(csv_path)
        
        # Apply filtering
        original_len = len(self.df)
        
        # Filter out unwanted events (e.g., DESTROY)
        if filter_events:
            self.df = self.df[~self.df['event'].isin(filter_events)]
            filtered_len = len(self.df)
            print(f"Dataset: Filtered out events {filter_events}: {original_len - filtered_len} rows removed ({100*(original_len-filtered_len)/original_len:.1f}%)")
        
        # Filter out unwanted entities (e.g., Sheep - captured, not built)
        if filter_entities:
            pre_filter_len = len(self.df)
            self.df = self.df[~self.df['entity'].astype(str).isin(filter_entities)]
            filtered_len = len(self.df)
            print(f"Dataset: Filtered out entities {filter_entities}: {pre_filter_len - filtered_len} rows removed ({100*(pre_filter_len-filtered_len)/pre_filter_len:.1f}%)")
        
        # Filter out events at timestamp 0 (outside player control - starting units, etc.)
        if 'time' in self.df.columns:
            pre_filter_len = len(self.df)
            self.df = self.df[self.df['time'] != 0]
            filtered_len = len(self.df)
            if pre_filter_len - filtered_len > 0:
                print(f"Dataset: Filtered out timestamp=0 events: {pre_filter_len - filtered_len} rows removed ({100*(pre_filter_len-filtered_len)/pre_filter_len:.1f}%)")
        
        self.max_len = max_len
        self.entity_vocab = entity_vocab
        self.event_vocab = event_vocab
        self.civ_vocab = civ_vocab
        self.map_vocab = map_vocab
        self.use_time_features = use_time_features
        self.truncation_strategy = truncation_strategy

        # Group by game_id and profile_id; maintain sorted order by time
        grouped = defaultdict(list)
        for _, row in self.df.iterrows():
            key = (row['game_id'], row['profile_id'])
            grouped[key].append(row)

        self.examples = []  # each is dict with tokens, times, label, civs
        for key, rows in grouped.items():
            # sort by `time`
            rows_sorted = sorted(rows, key=lambda r: r['time'])
            entities = [str(r['entity']) for r in rows_sorted]
            events = [str(r['event']) for r in rows_sorted]
            # Optional resource columns - fill with 0 if missing
            if 'wood' in self.df.columns:
                wood = [r['wood'] for r in rows_sorted]
            else:
                wood = [0 for _ in rows_sorted]

            if 'food' in self.df.columns:
                food = [r['food'] for r in rows_sorted]
            else:
                food = [0 for _ in rows_sorted]

            if 'stone' in self.df.columns:
                stone = [r['stone'] for r in rows_sorted]
            else:
                stone = [0 for _ in rows_sorted]

            if 'gold' in self.df.columns:
                gold = [r['gold'] for r in rows_sorted]
            else:
                gold = [0 for _ in rows_sorted] 
            # prefer delta_time_scaled if present and numeric, otherwise fall back to `time`
            times = []
            for r in rows_sorted:
                val = None
                #if 'delta_time_scaled' in r and not pd.isna(r['delta_time_scaled']):
                #    val = r['delta_time_scaled']
                #else:
                #    val = r['time']
                val = r['time']
                try:
                    times.append(float(val))
                except Exception:
                    # if conversion fails (e.g., unexpected string), fall back to 0.0
                    try:
                        times.append(float(str(val).strip()))
                    except Exception:
                        times.append(0.0)

            # player metadata from first row
            label = int(rows_sorted[0]['player_won'])
            player_civ = rows_sorted[0].get('player_civ')
            enemy_civ = rows_sorted[0].get('enemy_civ')
            # Some CSVs may not have a 'map' field or it may be NaN/None; normalize to UNK_TOKEN string
            raw_map = rows_sorted[0].get('map') if 'map' in rows_sorted[0].index else None
            if raw_map is None or (isinstance(raw_map, float) and pd.isna(raw_map)):
                game_map = UNK_TOKEN
            else:
                game_map = str(raw_map)
            game_id = rows_sorted[0]['game_id']

            self.examples.append({
                'game_id': game_id,
                'entities': entities,
                'events': events,
                'times': times,
                'wood': wood,
                'food': food,
                'gold': gold,
                'stone': stone,
                'map': game_map,
                'player_civ': player_civ,
                'enemy_civ': enemy_civ,
                'label': label
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # encode entity tokens
        ent_ids = [encode_token(e, self.entity_vocab) for e in ex['entities']]
        event_ids = [encode_token(ev, self.event_vocab) for ev in ex['events']]
        wood_arr = [w for w in ex['wood']]
        gold_arr = [w for w in ex['gold']]
        stone_arr = [w for w in ex['stone']]
        food_arr = [w for w in ex['food']]

        # truncate only if max_len is set and shorter than the sequence; otherwise keep all events
        if self.max_len is not None and len(ent_ids) > self.max_len:
            k = self.max_len
            if self.truncation_strategy == 'head':
                ent_ids = ent_ids[:k]
                event_ids = event_ids[:k]
                times = ex['times'][:k]
            elif self.truncation_strategy == 'tail':
                ent_ids = ent_ids[-k:]
                event_ids = event_ids[-k:]
                times = ex['times'][-k:]
            elif self.truncation_strategy == 'head_tail':
                h = k // 2
                ent_ids = ent_ids[:h] + ent_ids[-(k - h):]
                event_ids = event_ids[:h] + event_ids[-(k - h):]
                times = ex['times'][:h] + ex['times'][-(k - h):]
            else:
                raise ValueError(f"Unknown truncation_strategy: {self.truncation_strategy}")
        else:
            times = ex['times']

        # numeric features
        times_arr = np.array(times, dtype=np.float32)
        # normalize time by max time in sequence
        if len(times_arr) > 0:
            times_arr = times_arr / (times_arr.max() + 1e-6)
        else:
            times_arr = np.zeros((0,), dtype=np.float32)

        # map id: use UNK if mapping not found or map token missing
        map_token = ex.get('map', UNK_TOKEN)
        map_id = self.map_vocab.get(str(map_token), self.map_vocab.get(UNK_TOKEN, 1))

        sample = {
            'game_id': ex['game_id'],
            'entity_ids': torch.tensor(ent_ids, dtype=torch.long),
            'event_ids': torch.tensor(event_ids, dtype=torch.long),
            'wood': torch.tensor(wood_arr, dtype=torch.int),
            'stone': torch.tensor(stone_arr, dtype=torch.int),
            'gold': torch.tensor(gold_arr, dtype=torch.int),
            'food': torch.tensor(food_arr, dtype=torch.int),
            'times': torch.tensor(times_arr, dtype=torch.float32),
            'map': torch.tensor(map_id, dtype=torch.long),
            'player_civ': torch.tensor(self.civ_vocab.get(ex['player_civ'], self.civ_vocab.get(UNK_TOKEN, 1)), dtype=torch.long),
            'enemy_civ': torch.tensor(self.civ_vocab.get(ex['enemy_civ'], self.civ_vocab.get(UNK_TOKEN, 1)), dtype=torch.long),
            'label': torch.tensor(ex['label'], dtype=torch.float32)
        }
        return sample


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for AoEEventDataset.

    Takes a list of batch dictionaries returned by aoEEventDataset.__getitem__ and
    returns a single dictionary containing the following:
        - 'entity_ids': Padded tensor of shape (B, max_len) containing entity token ids.
        - 'event_ids': Padded tensor of shape (B, max_len) containing event token ids.
        - 'times': Padded tensor of shape (B, max_len) containing time values in seconds.
        - 'attention_mask': Padded tensor of shape (B, max_len) containing attention mask values.
        - 'map': Tensor of shape (B) containing map IDs.
        - 'player_civ': Tensor of shape (B) containing player civ IDs.
        - 'enemy_civ': Tensor of shape (B) containing enemy civ IDs.
        - 'labels': Tensor of shape (B) containing win probability labels.
    """
    batch_entity = [b['entity_ids'] for b in batch]
    batch_event = [b['event_ids'] for b in batch]
    batch_wood = [b['wood'] for b in batch]
    batch_food = [b['food'] for b in batch]
    batch_stone = [b['stone'] for b in batch]
    batch_gold = [b['gold'] for b in batch]
    batch_times = [b['times'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    maps = torch.stack([b['map'] for b in batch])
    civs = torch.stack([b['player_civ'] for b in batch])
    enemy_civs = torch.stack([b['enemy_civ'] for b in batch])

    max_len = max([t.size(0) for t in batch_entity])
    padded_entities = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_events = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_times = torch.zeros((len(batch), max_len), dtype=torch.float32)
    padded_wood = torch.zeros((len(batch), max_len), dtype=torch.int)
    padded_food = torch.zeros((len(batch), max_len), dtype=torch.int)
    padded_stone = torch.zeros((len(batch), max_len), dtype=torch.int)
    padded_gold = torch.zeros((len(batch), max_len), dtype=torch.int)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i in range(len(batch)):
        l = batch_entity[i].size(0)
        padded_entities[i, :l] = batch_entity[i]
        padded_events[i, :l] = batch_event[i]
        # padded_wood[i, :l] = batch_wood[i]
        # padded_food[i, :l] = batch_food[i]
        # padded_gold[i, :l] = batch_stone[i]
        # padded_stone[i, :l] = batch_gold[i]
        padded_times[i, :l] = batch_times[i]
        attention_mask[i, :l] = 1

    return {
        'entity_ids': padded_entities,
        'event_ids': padded_events,
        # 'wood': padded_wood,
        # 'stone': padded_stone,
        # 'food': padded_food,
        # 'gold': padded_gold,
        'times': padded_times,
        'attention_mask': attention_mask,
        'map': maps,
        'player_civ': civs,
        'enemy_civ': enemy_civs,
        'labels': labels
    }
