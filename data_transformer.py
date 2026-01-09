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
from typing import List, Dict, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocabs(df: pd.DataFrame, min_freq: int = 1) -> Dict[str, Dict[str, int]]:
   
    """
    Builds three vocabularies for "entity", "event" and "civ" columns in the given DataFrame.

    The vocabularies are built by counting the frequency of each token in the columns.
    Tokens with a frequency below the given min_freq are not included in the vocabularies.
    The vocabularies are returned as a dictionary with keys 'entity_vocab', 'event_vocab', and 'civ_vocab'.
    Each vocabulary is a dictionary mapping tokens to their indices.

    :param df: A DataFrame containing the columns 'entity', 'event', 'player_civ', and 'enemy_civ'.
    :param min_freq: The minimum frequency of a token to be included in the vocabularies.
    :return: A dictionary containing the built vocabularies.
    """
    entity_counts = Counter(df['entity'].astype(str).tolist())
    event_counts = Counter(df['event'].astype(str).tolist())
    civ_counts = Counter(df['player_civ'].astype(str).tolist() + df['enemy_civ'].astype(str).tolist())

    def make_vocab(counter: Counter) -> Dict[str, int]:
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for tok, cnt in counter.items():
            if cnt >= min_freq:
                vocab[tok] = len(vocab)
        return vocab

    return {
        'entity_vocab': make_vocab(entity_counts),
        'event_vocab': make_vocab(event_counts),
        'civ_vocab': make_vocab(civ_counts)
    }


def encode_token(token: str, vocab: Dict[str, int]) -> int:
    return vocab.get(token, vocab.get(UNK_TOKEN, 1))

class AoEEventDataset(Dataset):
    """Dataset of sequences per player-game.

    Each item corresponds to a single player's event sequence in a game, with label in `player_won`.
    """

    def __init__(self, csv_path: str, entity_vocab: Dict[str, int], event_vocab: Dict[str, int], civ_vocab: Dict[str, int], max_len: Optional[int] = None, use_time_features: bool = True, truncation_strategy: str = 'head_tail'):
        """
        Initialize the dataset from a CSV file containing event sequences per player-game.

        :param csv_path: Path to the CSV file containing the event sequences.
        :param entity_vocab: Vocabulary for "entity" tokens.
        :param event_vocab: Vocabulary for "event" tokens.
        :param civ_vocab: Vocabulary for "civ" tokens.
        :param max_len: Maximum length of a sequence. If None, do not truncate and use all events.
                        If set, sequences longer than `max_len` will be truncated according to
                        `truncation_strategy`.
        :param use_time_features: If True, use delta time scaled as a feature.
        :param truncation_strategy: One of {'head', 'tail', 'head_tail'} determining how to
                                    truncate sequences when they exceed `max_len`.
        """
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.entity_vocab = entity_vocab
        self.event_vocab = event_vocab
        self.civ_vocab = civ_vocab
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
            player_civ = rows_sorted[0]['player_civ']
            enemy_civ = rows_sorted[0]['enemy_civ']
            game_id = rows_sorted[0]['game_id']

            self.examples.append({
                'game_id': game_id,
                'entities': entities,
                'events': events,
                'times': times,
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

        sample = {
            'game_id': ex['game_id'],
            'entity_ids': torch.tensor(ent_ids, dtype=torch.long),
            'event_ids': torch.tensor(event_ids, dtype=torch.long),
            'times': torch.tensor(times_arr, dtype=torch.float32),
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
        - 'player_civ': Tensor of shape (B) containing player civ IDs.
        - 'enemy_civ': Tensor of shape (B) containing enemy civ IDs.
        - 'labels': Tensor of shape (B) containing win probability labels.
    """
    batch_entity = [b['entity_ids'] for b in batch]
    batch_event = [b['event_ids'] for b in batch]
    batch_times = [b['times'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    civs = torch.stack([b['player_civ'] for b in batch])
    enemy_civs = torch.stack([b['enemy_civ'] for b in batch])

    max_len = max([t.size(0) for t in batch_entity])
    padded_entities = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_events = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_times = torch.zeros((len(batch), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i in range(len(batch)):
        l = batch_entity[i].size(0)
        padded_entities[i, :l] = batch_entity[i]
        padded_events[i, :l] = batch_event[i]
        padded_times[i, :l] = batch_times[i]
        attention_mask[i, :l] = 1

    return {
        'entity_ids': padded_entities,
        'event_ids': padded_events,
        'times': padded_times,
        'attention_mask': attention_mask,
        'player_civ': civs,
        'enemy_civ': enemy_civs,
        'labels': labels
    }
