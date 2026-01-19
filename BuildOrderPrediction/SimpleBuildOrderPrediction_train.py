import torch
import torch.nn as nn
import math
import os
import argparse
import random
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import wandb
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class SequencePredictor(nn.Module):
    """
    Transformer-based sequence predictor for entity build orders.
    Predicts the next entity in a sequence (focuses on build order placement).
    Conditioned on player civilization, enemy civilization, and map.
    """
    
    def __init__(self,
                 vocab_size_entity: int,
                 civ_vocab_size: int,
                 map_vocab_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1. Entity embedding (focus on build order, not events)
        self.entity_embed = nn.Embedding(vocab_size_entity, d_model)
        
        # 2. Condition embeddings (player civ, enemy civ, map)
        self.civ_embed = nn.Embedding(civ_vocab_size, d_model)
        self.map_embed = nn.Embedding(map_vocab_size, d_model)
        self.condition_proj = nn.Linear(d_model * 3, d_model)  # player_civ + enemy_civ + map
        
        # 3. Positional encoding
        # Allow for extra tokens (BOS, condition embedding) by adding margin
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 32)
        
        # 4. Transformer encoder for sequence understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 5. Output head for entity prediction only
        self.entity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size_entity)
        )
        
        # 6. Start tokens (learnable)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Mask value for invalid positions
        self.mask_value = float(-1e4)
    
    def encode_condition(self, player_civ: torch.Tensor, enemy_civ: torch.Tensor, 
                         map_id: torch.Tensor) -> torch.Tensor:
        """
        Encode conditioning features: player civ + enemy civ + map.
        
        Args:
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            map_id: (B,) map IDs
            
        Returns:
            condition: (B, 1, d_model) condition embedding
        """
        p_civ = self.civ_embed(player_civ)   # (B, d_model)
        e_civ = self.civ_embed(enemy_civ)    # (B, d_model)
        map_emb = self.map_embed(map_id)     # (B, d_model)
        
        # Concatenate and project
        condition = torch.cat([p_civ, e_civ, map_emb], dim=-1)  # (B, d_model*3)
        condition = self.condition_proj(condition)  # (B, d_model)
        
        return condition.unsqueeze(1)  # (B, 1, d_model)
        
    def create_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Create embeddings for entity sequence (build order).
        
        Args:
            entity_ids: (B, L) entity token ids
            
        Returns:
            entity_embeddings: (B, L, d_model)
        """
        return self.entity_embed(entity_ids)  # (B, L, d_model)
    
    def forward(self,
                entity_sequence: torch.Tensor,
                player_civ: torch.Tensor,
                enemy_civ: torch.Tensor,
                map_id: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                predict_next: bool = True) -> torch.Tensor:
        """
        Forward pass for entity sequence prediction (build order).
        
        Args:
            entity_sequence: (B, L) entity token ids (padded)
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            map_id: (B,) map IDs
            attention_mask: (B, L) boolean mask (1 for valid positions)
            predict_next: if True, predict next element; if False, predict all elements
            
        Returns:
            entity_logits: (B, vocab_size_entity) or (B, L, vocab_size_entity)
        """
        batch_size, seq_len = entity_sequence.shape
        device = entity_sequence.device
        
        # Encode condition (player civ, enemy civ, map)
        condition = self.encode_condition(player_civ, enemy_civ, map_id)  # (B, 1, d_model)
        
        # Create entity embeddings for sequence
        entity_emb = self.create_entity_embeddings(entity_sequence)  # (B, L, d_model)
        
        # Prepend condition to sequence
        combined_emb = torch.cat([condition, entity_emb], dim=1)  # (B, 1+L, d_model)
        
        # Add positional encoding
        combined_emb = self.pos_encoder(combined_emb)
        
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = (entity_sequence != 0)  # Assuming 0 is padding token
        
        # Prepend True (valid) for condition token
        condition_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        attention_mask = torch.cat([condition_mask, attention_mask], dim=1)  # (B, 1+L)
        
        # Convert to transformer format (True positions are allowed, False positions are masked)
        src_key_padding_mask = ~attention_mask  # Invert for transformer
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(
            combined_emb,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, 1+L, d_model)
        
        # Remove condition token from output (keep only sequence positions)
        encoded = encoded[:, 1:, :]  # (B, L, d_model)
        
        if predict_next:
            # Predict only the next element after the sequence
            # Use the last valid position (from original mask, not including condition)
            original_mask = attention_mask[:, 1:]  # Remove condition from mask
            last_valid_indices = original_mask.sum(dim=1) - 1  # (B,)
            
            # Gather encoded states at last valid positions
            batch_indices = torch.arange(batch_size, device=device)
            last_hidden = encoded[batch_indices, last_valid_indices]  # (B, d_model)
            
            # Predict next entity
            entity_logits = self.entity_head(last_hidden)  # (B, vocab_size_entity)
            
        else:
            # Predict next element for each position (teacher forcing style)
            entity_logits = self.entity_head(encoded)  # (B, L, vocab_size_entity)
        
        return entity_logits
    
    def generate(self,
                 entity_sequence: torch.Tensor,
                 player_civ: torch.Tensor,
                 enemy_civ: torch.Tensor,
                 map_id: torch.Tensor,
                 max_new_tokens: int = 10,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate new entity sequence autoregressively (build order).
        
        Args:
            entity_sequence: (B, L) initial entity sequence
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            map_id: (B,) map IDs
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: if not None, use top-k sampling
            
        Returns:
            generated_entities: (B, L + max_new_tokens)
        """
        batch_size = entity_sequence.shape[0]
        device = entity_sequence.device
        
        # Initialize with input sequence
        current_entities = entity_sequence.clone()
        
        for _ in range(max_new_tokens):
            # Get predictions for next element
            with torch.no_grad():
                entity_logits = self.forward(
                    current_entities,
                    player_civ,
                    enemy_civ,
                    map_id,
                    predict_next=True
                )
            
            # Apply temperature
            entity_logits = entity_logits / temperature
            
            # Apply top-k if specified
            if top_k is not None:
                entity_logits = self._top_k_filtering(entity_logits, top_k)
            
            # Sample from distribution
            entity_probs = torch.softmax(entity_logits, dim=-1)
            next_entity = torch.multinomial(entity_probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            current_entities = torch.cat([current_entities, next_entity], dim=1)
        
        return current_entities
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply top-k filtering to logits.
        """
        values, _ = torch.topk(logits, k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SequencePredictorTrainer:
    """
    Trainer for the SequencePredictor model.
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 use_amp: bool = False,
                 grad_accum_steps: int = 1,
                 civ_entity_mask: Optional[torch.Tensor] = None,
                 entity_class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Civ-entity mask: (num_civs, vocab_size_entity)
        # True means the civ can build that entity
        self.civ_entity_mask = civ_entity_mask
        
        # Entity class weights for balanced loss
        self.entity_class_weights = entity_class_weights
        if self.entity_class_weights is not None:
            self.entity_class_weights = self.entity_class_weights.to(device)
            
        self.label_smoothing = label_smoothing
        
        # AMP setup
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        self.grad_accum_steps = grad_accum_steps
        self._accum_step = 0
        
    def compute_loss(self,
                entity_logits: torch.Tensor,
                entity_targets: torch.Tensor,
                player_civ: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for entity sequence prediction with masked label smoothing.
        
        Args:
            entity_logits: (B, vocab_size_entity) or (B, L, vocab_size_entity)
            entity_targets: (B,) or (B, L) entity targets
            player_civ: (B,) player civilization IDs for civ-entity masking
            mask: optional mask for valid positions
            
        Returns:
            loss, loss_dict
        """
        # Apply civ-entity mask if available
        batch_mask = None
        if self.civ_entity_mask is not None and player_civ is not None:
            # Get valid entity mask for each sample's civ
            batch_mask = self.civ_entity_mask[player_civ]  # (B, vocab_size)
            
            if entity_logits.dim() == 3:  # (B, L, vocab_size) - teacher forcing
                # Expand mask for sequence dimension
                B, L, V = entity_logits.shape
                batch_mask = batch_mask.unsqueeze(1).expand(B, L, V)
        
        # Mask logits with -inf (standard masking)
        # We rely on custom loss to handle -inf correctly via proper target masking
        if batch_mask is not None:
            entity_logits = entity_logits.masked_fill(~batch_mask, float('-inf'))
            
        # Get class weights
        weights = self.entity_class_weights
        
        # Custom Loss Calculation
        if self.label_smoothing > 0:
            # Compute log probabilities
            log_probs = F.log_softmax(entity_logits, dim=-1)
            
            # Construct smoothed targets
            with torch.no_grad():
                epsilon = self.label_smoothing
                
                # Create base distribution
                if batch_mask is not None:
                    # Distribute epsilon over all VALID classes
                    valid_count = batch_mask.sum(dim=-1, keepdim=True)
                    # Avoid division by zero (should not happen if mask has at least 1 valid)
                    valid_count = torch.clamp(valid_count, min=1.0)
                    smooth_val = epsilon / valid_count
                    
                    true_dist = torch.zeros_like(log_probs)
                    true_dist = torch.where(batch_mask, smooth_val, true_dist)
                else:
                    V = log_probs.size(-1)
                    true_dist = torch.full_like(log_probs, epsilon / V)
                
                # Add (1 - epsilon) to the correct target class
                # This formulation assumes: target dist = (1-eps)*one_hot + eps*uniform_over_valid
                # Sum of prob = (1-eps) + eps = 1.0 (Correct)
                
                # Gather targets
                if entity_logits.dim() == 2:
                    current_targets = entity_targets.unsqueeze(-1)
                else:
                    # Flatten for easier handling or keep 3D
                    current_targets = entity_targets.unsqueeze(-1)
                
                # Scatter add
                src = torch.full_like(current_targets, 1.0 - epsilon, dtype=true_dist.dtype)
                true_dist.scatter_add_(-1, current_targets, src)
            
            # Compute cross entropy: - sum(target * log_prob)
            # Handle -inf in log_probs by masking the product
            # If target is 0 (invalid class), the term should be 0.
            # But 0 * -inf is NaN. So we must ensure log_probs is finite or use masking.
            
            # Zero out log_probs where true_dist is 0 (classes with no probability mass)
            # This prevents 0 * -inf = nan issues while keeping valid contributions
            # Note: We use true_dist > 0 instead of batch_mask because the target class
            # must always contribute, even if it's a masked civ entity (the target is ground truth)
            safe_log_probs = torch.where(true_dist > 0, log_probs, torch.zeros_like(log_probs))
                
            # Sum over classes
            loss_per_sample = -torch.sum(true_dist * safe_log_probs, dim=-1)
            
            # Apply class weights if provided (weight based on target class)
            if weights is not None:
                # Flatten weights if needed
                weights = weights.to(loss_per_sample.device)
                sample_weights = weights[entity_targets]
                loss_per_sample = loss_per_sample * sample_weights
                
            entity_loss = loss_per_sample.mean()
            
        else:
            # Standard Cross Entropy
            if entity_logits.dim() == 2:
                entity_loss = F.cross_entropy(entity_logits, entity_targets, weight=weights)
            else:
                # (B, L, V) -> (B, V, L) for cross_entropy
                entity_loss = F.cross_entropy(entity_logits.transpose(1, 2), entity_targets, weight=weights)
        
        loss_dict = {
            'loss': entity_loss.item(),
        }
        
        return entity_loss, loss_dict
    
    def train_step(self,
              batch: Dict[str, torch.Tensor],
              teacher_forcing_ratio: float = 0.5) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: dictionary with 'entity_ids' and optionally 'mask'
            teacher_forcing_ratio: probability of using teacher forcing
            
        Returns:
            loss_dict
        """
        self.model.train()
        
        # Move batch to device
        entity_ids = batch['entity_ids'].to(self.device)
        player_civ = batch['player_civ'].to(self.device)
        enemy_civ = batch['enemy_civ'].to(self.device)
        map_id = batch['map_id'].to(self.device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Prepare targets (shifted by one for next element prediction)
        if teacher_forcing_ratio > 0 and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: predict all next elements
            # Use .contiguous() to ensure tensors are contiguous in memory
            entity_targets = entity_ids[:, 1:].contiguous()
            
            # Input sequences (excluding last element)
            entity_input = entity_ids[:, :-1].contiguous()
            
            # Adjust mask if provided
            if mask is not None:
                input_mask = mask[:, :-1].contiguous()
            else:
                input_mask = None
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    entity_logits = self.model(
                        entity_input, player_civ, enemy_civ, map_id, input_mask, predict_next=False
                    )
            else:
                entity_logits = self.model(
                    entity_input, player_civ, enemy_civ, map_id, input_mask, predict_next=False
                )
                
        else:
            # Standard next element prediction
            # For each sequence, predict the element after the last valid position
            if mask is not None:
                # Find last valid positions
                seq_lengths = mask.sum(dim=1).long() - 1  # Index of last valid element
                
                # Clamp seq_lengths to prevent out-of-bounds access when fetching targets
                # seq_lengths + 1 must be < entity_ids.size(1), so seq_lengths must be < entity_ids.size(1) - 1
                max_valid_idx = entity_ids.size(1) - 2  # Maximum valid value for seq_lengths
                seq_lengths = torch.clamp(seq_lengths, min=0, max=max_valid_idx)
                
                # Create input sequences (all but last element)
                batch_indices = torch.arange(entity_ids.size(0), device=self.device)
                
                # Use torch.narrow or slicing with .contiguous()
                max_length = seq_lengths.max().item() + 1  # +1 because we want to include up to seq_lengths
                entity_input = entity_ids[:, :max_length].contiguous()
                
                # Targets are the elements after last valid positions (now safe due to clamping)
                entity_targets = entity_ids[batch_indices, seq_lengths + 1].contiguous()
                
               # Vectorized mask creation
                batch_size, max_len = entity_input.shape

                # Create input mask - positions < seq_lengths are valid
                input_mask = (torch.arange(max_len, device=self.device).unsqueeze(0).expand(batch_size, -1) 
                              < seq_lengths.unsqueeze(1))
            else:
                # If no mask, assume all positions except last are valid
                entity_input = entity_ids[:, :-1].contiguous()
                entity_targets = entity_ids[:, -1].contiguous()
                input_mask = None
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    entity_logits = self.model(
                        entity_input, player_civ, enemy_civ, map_id, input_mask, predict_next=True
                    )
            else:
                entity_logits = self.model(
                    entity_input, player_civ, enemy_civ, map_id, input_mask, predict_next=True
                )
        
        # Compute loss (with civ-entity masking)
        loss, loss_dict = self.compute_loss(
            entity_logits, entity_targets, player_civ, mask
        )
        
        # Normalize loss for gradient accumulation
        loss = loss / self.grad_accum_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        self._accum_step += 1
        if self._accum_step >= self.grad_accum_steps:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_step = 0
        
        return loss_dict
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            dataloader: validation/test dataloader
            
        Returns:
            metrics dictionary
        """
        self.model.eval()
        total_loss = 0.0
        total_correct_entity = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                entity_ids = batch['entity_ids'].to(self.device)
                player_civ = batch['player_civ'].to(self.device)
                enemy_civ = batch['enemy_civ'].to(self.device)
                map_id = batch['map_id'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                
                # Use all but last element as input, last element as target
                # Make tensors contiguous
                entity_input = entity_ids[:, :-1].contiguous()
                entity_targets = entity_ids[:, -1].contiguous()
                
                if mask is not None:
                    input_mask = mask[:, :-1].contiguous()
                else:
                    input_mask = None
                
                # Forward pass
                entity_logits = self.model(
                    entity_input, player_civ, enemy_civ, map_id, input_mask, predict_next=True
                )
                
                # Compute loss (with civ-entity masking)
                loss, _ = self.compute_loss(
                    entity_logits,
                    entity_targets,
                    player_civ
                )
                
                # Compute accuracy
                entity_preds = torch.argmax(entity_logits, dim=-1)
                
                total_loss += loss.item() * entity_ids.size(0)
                total_correct_entity += (entity_preds == entity_targets).sum().item()
                total_samples += entity_ids.size(0)
        
        avg_loss = total_loss / total_samples
        entity_accuracy = total_correct_entity / total_samples
        
        return {
            'loss': avg_loss,
            'entity_accuracy': entity_accuracy
        }


class SequenceDataset(Dataset):
    """Dataset for entity sequence prediction with game-level splitting support."""
    def __init__(self, csv_path: str, entity_vocab: Dict[str, int], 
                 civ_vocab: Dict[str, int],
                 map_vocab: Dict[str, int], max_seq_len: int = 100,
                 filter_events: List[str] = None, filter_entities: List[str] = None,
                 only_game_start: bool = True):
        """
        Args:
            csv_path: Path to CSV file
            entity_vocab: Vocabulary dictionary for entities
            civ_vocab: Vocabulary dictionary for civilizations
            map_vocab: Vocabulary dictionary for maps
            max_seq_len: Maximum sequence length
            filter_events: List of event types to EXCLUDE (e.g., ['DESTROY'])
            filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep'])
            only_game_start: If True, only use the first chunk of each game (start of build order)
        """
        import pandas as pd
        
        self.df = pd.read_csv(csv_path)
        self.entity_vocab = entity_vocab
        self.civ_vocab = civ_vocab
        self.map_vocab = map_vocab
        self.max_seq_len = max_seq_len
        
        original_len = len(self.df)
        
        # Filter out unwanted events (e.g., DESTROY)
        if filter_events:
            self.df = self.df[~self.df['event'].isin(filter_events)]
            filtered_len = len(self.df)
            print(f"Filtered out events {filter_events}: {original_len - filtered_len} rows removed ({100*(original_len-filtered_len)/original_len:.1f}%)")
        
        # Filter out unwanted entities (e.g., Sheep - captured, not built)
        if filter_entities:
            pre_filter_len = len(self.df)
            self.df = self.df[~self.df['entity'].astype(str).isin(filter_entities)]
            filtered_len = len(self.df)
            print(f"Filtered out entities {filter_entities}: {pre_filter_len - filtered_len} rows removed ({100*(pre_filter_len-filtered_len)/pre_filter_len:.1f}%)")
        
        # Filter out events at timestamp 0 (outside player control - starting units, etc.)
        if 'time' in self.df.columns:
            pre_filter_len = len(self.df)
            self.df = self.df[self.df['time'] != 0]
            filtered_len = len(self.df)
            if pre_filter_len - filtered_len > 0:
                print(f"Filtered out timestamp=0 events: {pre_filter_len - filtered_len} rows removed ({100*(pre_filter_len-filtered_len)/pre_filter_len:.1f}%)")
        
        # Check if map column exists
        self.has_map = 'map' in self.df.columns
        
        # Group by game_id and profile_id
        self.sequences = []
        self.game_to_indices = {}  # Maps game_id to list of indices in self.sequences
        
        grouped = self.df.groupby(['game_id', 'profile_id'])
        
        for (game_id, profile_id), group in grouped:
            group = group.sort_values('time')
            entities = [entity_vocab.get(str(e), 1) for e in group['entity'].tolist()]  # 1 for UNK
            
            # Skip empty sequences (can happen after filtering)
            if len(entities) == 0:
                continue
            
            # Get condition info (same for all rows in this group)
            first_row = group.iloc[0]
            player_civ = civ_vocab.get(str(first_row.get('player_civ', '<UNK>')), 1)
            enemy_civ = civ_vocab.get(str(first_row.get('enemy_civ', '<UNK>')), 1)
            map_id = map_vocab.get(str(first_row.get('map', '<UNK>')), 1) if self.has_map else 1
            
            # Split into sequences of max_seq_len
            # If only_game_start is True, only use the first chunk
            chunks_to_use = 1 if only_game_start else (len(entities) + max_seq_len - 1) // max_seq_len
            
            for chunk_idx in range(chunks_to_use):
                i = chunk_idx * max_seq_len
                seq_entities = entities[i:i + max_seq_len]
                
                # Prepend <BOS> token (id=2) to signal game start
                # This is only prepended for the first chunk of each game
                if chunk_idx == 0:
                    seq_entities = [2] + seq_entities  # 2 = <BOS> token
                
                actual_len = len(seq_entities)
                
                # Pad if necessary (account for +1 max_seq_len since BOS is prepended)
                target_len = max_seq_len + 1 if chunk_idx == 0 else max_seq_len
                if actual_len < target_len:
                    pad_len = target_len - actual_len
                    seq_entities = seq_entities + [0] * pad_len  # 0 for PAD
                
                seq_idx = len(self.sequences)
                self.sequences.append({
                    'entity_ids': seq_entities,
                    'mask': [1] * actual_len + [0] * (len(seq_entities) - actual_len),
                    'game_id': game_id,
                    'player_civ': player_civ,
                    'enemy_civ': enemy_civ,
                    'map_id': map_id
                })
                
                # Map game_id to sequence index
                if game_id not in self.game_to_indices:
                    self.game_to_indices[game_id] = []
                self.game_to_indices[game_id].append(seq_idx)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'entity_ids': torch.tensor(seq['entity_ids'], dtype=torch.long),
            'mask': torch.tensor(seq['mask'], dtype=torch.bool),
            'game_id': seq['game_id'],
            'player_civ': torch.tensor(seq['player_civ'], dtype=torch.long),
            'enemy_civ': torch.tensor(seq['enemy_civ'], dtype=torch.long),
            'map_id': torch.tensor(seq['map_id'], dtype=torch.long)
        }
    
    def get_game_ids(self):
        """Get list of unique game IDs."""
        return list(self.game_to_indices.keys())


def create_data_loaders(csv_path: str, entity_vocab: Dict[str, int], 
                       civ_vocab: Dict[str, int],
                       map_vocab: Dict[str, int], batch_size: int = 32,
                       max_seq_len: int = 100, val_split: float = 0.2,
                       seed: int = 42, filter_events: List[str] = None,
                       filter_entities: List[str] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with game-level splitting.
    
    Args:
        csv_path: Path to CSV file
        entity_vocab: Vocabulary dictionary for entities
        civ_vocab: Vocabulary dictionary for civilizations
        map_vocab: Vocabulary dictionary for maps
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        val_split: Validation split ratio
        seed: Random seed for reproducibility
        filter_events: List of event types to EXCLUDE (e.g., ['DESTROY'])
        filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep'])
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    dataset = SequenceDataset(csv_path, entity_vocab, civ_vocab, map_vocab, 
                              max_seq_len, filter_events=filter_events,
                              filter_entities=filter_entities)
    
    # Split by game ID
    all_game_ids = dataset.get_game_ids()
    train_games, val_games = train_test_split(
        all_game_ids, test_size=val_split, random_state=seed
    )
    
    # Get indices for train and validation
    train_indices = []
    for game_id in train_games:
        train_indices.extend(dataset.game_to_indices[game_id])
    
    val_indices = []
    for game_id in val_games:
        val_indices.extend(dataset.game_to_indices[game_id])
    
    # Create subsets
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train set: {len(train_ds)} sequences from {len(train_games)} games")
    print(f"Val set: {len(val_ds)} sequences from {len(val_games)} games")
    
    return train_loader, val_loader


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Sequence Predictor')
    parser.add_argument('--csv_path', type=str, default='input_with_map.csv')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                       help='Teacher forcing ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # WandB arguments
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity')
    parser.add_argument('--wandb_project', type=str, default='DeepLearning-SimpleBuildOrder',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Exclude DESTROY events - we only care about BUILD/FINISH for build orders
    filter_events = ['DESTROY']
    # Exclude Sheep - they are captured units, not built, so no strategy involved
    filter_entities = ['Sheep']
    
    # Build vocabularies from data
    print("Building vocabularies...")
    entity_vocab, civ_vocab, map_vocab = build_vocabularies(args.csv_path, filter_events=filter_events, filter_entities=filter_entities)
    print(f"Entity vocabulary size: {len(entity_vocab)}")
    print(f"Civilization vocabulary size: {len(civ_vocab)}")
    print(f"Map vocabulary size: {len(map_vocab)}")
    
    # Create data loaders with game-level splitting
    print("Creating data loaders with game-level splitting...")
    train_loader, val_loader = create_data_loaders(
        csv_path=args.csv_path,
        entity_vocab=entity_vocab,
        civ_vocab=civ_vocab,
        map_vocab=map_vocab,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        val_split=args.val_split,
        seed=args.seed,
        filter_events=filter_events,
        filter_entities=filter_entities
    )
    
    # Create model
    print("Initializing model...")
    model = SequencePredictor(
        vocab_size_entity=len(entity_vocab),
        civ_vocab_size=len(civ_vocab),
        map_vocab_size=len(map_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    # Build civ-entity mapping for masking invalid entity predictions
    print("Building civ-entity mapping...")
    civ_entity_mapping = build_civ_entity_mapping(args.csv_path, filter_events=filter_events, filter_entities=filter_entities)
    
    # Convert mapping to mask tensor
    print("Creating civ-entity mask tensor...")
    civ_entity_mask = create_civ_entity_mask(
        civ_entity_mapping, entity_vocab, civ_vocab, device
    )
    
    # Compute entity class weights for balanced loss
    print("Computing entity class weights...")
    entity_class_weights = compute_entity_class_weights(
        args.csv_path, entity_vocab, filter_events=filter_events, filter_entities=filter_entities
    )
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Create trainer with civ-entity mask and class weights
    trainer = SequencePredictorTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_amp=True,  # Use mixed precision if available
        grad_accum_steps=args.grad_accum_steps,
        civ_entity_mask=civ_entity_mask,
        entity_class_weights=entity_class_weights,
        label_smoothing=args.label_smoothing
    )
    
    # Initialize Weights & Biases
    if not args.no_wandb:
        wb_cfg = {
            'learning_rate': args.lr,
            'architecture': 'SequencePredictor',
            'dataset': os.path.basename(args.csv_path),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'max_seq_len': args.max_seq_len,
            'dropout': args.dropout,
            'teacher_forcing_ratio': args.teacher_forcing_ratio,
            'grad_accum_steps': args.grad_accum_steps,
            'val_split': args.val_split,
            'entity_vocab_size': len(entity_vocab),
            'civ_vocab_size': len(civ_vocab),
            'map_vocab_size': len(map_vocab)
        }
        
        try:
            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=wb_cfg
            )
            wb_cfg = wandb.config
            print(f"W&B initialized. Run: {run.name}")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            args.no_wandb = True
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss_dict = trainer.train_step(
                batch,
                teacher_forcing_ratio=args.teacher_forcing_ratio
            )
            
            train_losses.append(loss_dict['loss'])
            
            # Log step-level training loss to W&B
            global_step = epoch * len(train_loader) + batch_idx
            if not args.no_wandb:
                wandb.log({
                    'step': global_step,
                    'train_step/loss': loss_dict['loss'],
                }, step=global_step)
            
            # Log batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss_dict['loss']:.4f}")
        
        # Calculate training metrics
        avg_train_loss = np.mean(train_losses)
        
        print(f"\nTraining Results:")
        print(f"  Average Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("\nRunning validation...")
        val_metrics = trainer.evaluate(val_loader)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Entity Accuracy: {val_metrics['entity_accuracy']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['entity_accuracy'],
                'entity_vocab': entity_vocab,
                'civ_vocab': civ_vocab,
                'map_vocab': map_vocab,
                # Save civ-entity mapping (convert sets to lists for serialization)
                'civ_entity_mapping': {k: list(v) for k, v in civ_entity_mapping.items()},
                'args': vars(args)
            }
            
            torch.save(checkpoint, 'best_model.pth')
            print(f"  ✓ Saved best model (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        # Log to Weights & Biases
        if not args.no_wandb:
            # Calculate gradient norm for monitoring
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            wandb.log({
                'epoch': epoch + 1,
                # Training metrics
                'train/loss': avg_train_loss,
                # Validation metrics
                'val/loss': val_metrics['loss'],
                'val/entity_accuracy': val_metrics['entity_accuracy'],
                # Training dynamics
                'training/learning_rate': current_lr,
                'training/grad_norm': total_norm,
                'training/patience_counter': patience_counter,
                # Best metrics tracking
                'best/val_loss': best_val_loss,
                'analysis/train_val_loss_gap': avg_train_loss - val_metrics['loss'],
            })
            
            # Log example predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_example_predictions(model, val_loader, entity_vocab, device, wandb)
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = trainer.evaluate(val_loader)
    print(f"\nFinal Validation Results (best model):")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Entity Accuracy: {final_metrics['entity_accuracy']:.4f}")
    
    # Save final model with additional metadata
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'entity_vocab': entity_vocab,
        'civ_vocab': civ_vocab,
        'map_vocab': map_vocab,
        # Save civ-entity mapping (convert sets to lists for serialization)
        'civ_entity_mapping': {k: list(v) for k, v in civ_entity_mapping.items()},
        'args': vars(args),
        'val_metrics': final_metrics,
        'best_val_loss': best_val_loss
    }
    torch.save(final_checkpoint, 'final_model.pth')
    
    # Log final results to wandb
    if not args.no_wandb:
        wandb.log({
            'final/val_loss': final_metrics['loss'],
            'final/entity_accuracy': final_metrics['entity_accuracy']
        })
        
        # Save model to wandb
        wandb.save('best_model.pth')
        wandb.save('final_model.pth')
        
        # Finish wandb run
        wandb.finish()
    
    print("\nModel saved as 'final_model.pth'")


def build_vocabularies(csv_path: str, filter_events: List[str] = None,
                       filter_entities: List[str] = None) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Build vocabulary dictionaries from CSV data.
    
    Args:
        csv_path: Path to CSV file
        filter_events: List of event types to EXCLUDE (e.g., ['DESTROY'])
        filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep'])
    
    Returns:
        entity_vocab, civ_vocab, map_vocab
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Filter out unwanted events before building vocabulary
    if filter_events:
        original_len = len(df)
        df = df[~df['event'].isin(filter_events)]
        print(f"Vocabularies: Filtered out events {filter_events}: {original_len - len(df)} rows excluded")
    
    # Filter out unwanted entities (e.g., Sheep - captured, not built)
    if filter_entities:
        pre_filter_len = len(df)
        df = df[~df['entity'].astype(str).isin(filter_entities)]
        print(f"Vocabularies: Filtered out entities {filter_entities}: {pre_filter_len - len(df)} rows excluded")
    
    # Filter out events at timestamp 0 (outside player control)
    if 'time' in df.columns:
        pre_filter_len = len(df)
        df = df[df['time'] != 0]
        if pre_filter_len - len(df) > 0:
            print(f"Vocabularies: Filtered out timestamp=0 events: {pre_filter_len - len(df)} rows excluded")
    
    # Get unique entities
    unique_entities = df['entity'].astype(str).unique()
    
    # Get unique civilizations (from both player and enemy columns)
    unique_civs = set()
    if 'player_civ' in df.columns:
        unique_civs.update(df['player_civ'].astype(str).unique())
    if 'enemy_civ' in df.columns:
        unique_civs.update(df['enemy_civ'].astype(str).unique())
    unique_civs = sorted(unique_civs)  # Sort for deterministic ordering
    
    # Get unique maps (if column exists)
    if 'map' in df.columns:
        unique_maps = sorted(df['map'].astype(str).unique())
    else:
        unique_maps = []
        print("Warning: 'map' column not found in CSV. Using placeholder map vocab.")
    
    # Create vocabularies with special tokens
    # <BOS> = Beginning-of-Sequence token to signal game start
    entity_vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        **{str(entity): i + 3 for i, entity in enumerate(unique_entities)}
    }
    
    civ_vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        **{str(civ): i + 2 for i, civ in enumerate(unique_civs)}
    }
    
    map_vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        **{str(m): i + 2 for i, m in enumerate(unique_maps)}
    }
    
    return entity_vocab, civ_vocab, map_vocab


def compute_entity_class_weights(csv_path: str, entity_vocab: Dict[str, int], 
                                  filter_events: List[str] = None,
                                  filter_entities: List[str] = None,
                                  smoothing: float = 0.05,
                                  high_importance_entities: List[str] = None,
                                  importance_boost: float = 5.0,
                                  low_importance_entities: List[str] = None,
                                  penalty_factor: float = 0.2) -> torch.Tensor:
    """Compute inverse frequency class weights for entity loss.
    
    Uses smoothed inverse frequency: weight = 1 / (count + smoothing * max_count)
    Then normalizes so mean weight = 1.0
    
    Additionally applies an importance boost multiplier to strategically important
    entities like age-up events.
    
    Args:
        csv_path: Path to CSV file
        entity_vocab: Entity vocabulary (token -> id)
        filter_events: List of event types to EXCLUDE
        filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep'])
        smoothing: Smoothing factor to prevent extreme weights for rare classes
        high_importance_entities: List of entity names that should receive extra weight
                                  (default: age-up entities)
        importance_boost: Multiplier for high importance entities (default: 5.0)
        low_importance_entities: List of entity names that should receive reduced weight
                                 (default: ['House'])
        penalty_factor: Multiplier for low importance entities (default: 0.2)
        
    Returns:
        weights: (vocab_size,) tensor of class weights
    """
    import pandas as pd
    
    # Default high-importance entities: age-up events are critical strategic decisions
    if high_importance_entities is None:
        high_importance_entities = [
            'Age Display Persistent 2',  # Age up to Feudal
            'Age Display Persistent 3',  # Age up to Castle  
            'Age Display Persistent 4',  # Age up to Imperial
        ]
        
    if low_importance_entities is None:
        low_importance_entities = ['House']
    
    df = pd.read_csv(csv_path)
    
    # Apply same filtering as training
    if filter_events:
        df = df[~df['event'].isin(filter_events)]
    
    # Filter out unwanted entities
    if filter_entities:
        df = df[~df['entity'].astype(str).isin(filter_entities)]
    
    # Filter out events at timestamp 0 (outside player control)
    if 'time' in df.columns:
        df = df[df['time'] != 0]
    
    # Count entity frequencies
    entity_counts = df['entity'].astype(str).value_counts()
    
    # Create weight tensor
    vocab_size = len(entity_vocab)
    weights = torch.ones(vocab_size)
    
    # Get max count for smoothing
    max_count = entity_counts.max()
    
    # Compute inverse frequency weights
    for entity_name, entity_id in entity_vocab.items():
        if entity_name in ('<PAD>', '<UNK>'):
            weights[entity_id] = 0.0  # Ignore special tokens
        elif entity_name in entity_counts.index:
            count = entity_counts[entity_name]
            # Smoothed inverse frequency
            weights[entity_id] = 1.0 / (count + smoothing * max_count)
        else:
            weights[entity_id] = 1.0  # Default weight for unseen entities
    
    # Normalize so mean weight (excluding special tokens) = 1.0
    valid_mask = weights > 0
    if valid_mask.sum() > 0:
        mean_weight = weights[valid_mask].mean()
        weights[valid_mask] = weights[valid_mask] / mean_weight
    
    # Apply importance boost to high-importance entities AFTER normalization
    boosted_entities = []
    for entity_name in high_importance_entities:
        if entity_name in entity_vocab:
            entity_id = entity_vocab[entity_name]
            old_weight = weights[entity_id].item()
            weights[entity_id] = weights[entity_id] * importance_boost
            boosted_entities.append((entity_name, old_weight, weights[entity_id].item()))
            
    # Apply penalty to low-importance entities
    penalized_entities = []
    for entity_name in low_importance_entities:
        if entity_name in entity_vocab:
            entity_id = entity_vocab[entity_name]
            old_weight = weights[entity_id].item()
            weights[entity_id] = weights[entity_id] * penalty_factor
            penalized_entities.append((entity_name, old_weight, weights[entity_id].item()))
    
    # Print summary
    print(f"Entity class weights computed:")
    print(f"  Min weight: {weights[valid_mask].min():.4f}")
    print(f"  Max weight: {weights[valid_mask].max():.4f}")
    print(f"  Mean weight: {weights[valid_mask].mean():.4f}")
    
    # Show weights for common entities
    inv_vocab = {v: k for k, v in entity_vocab.items()}
    weight_list = [(inv_vocab[i], weights[i].item()) for i in range(vocab_size) if weights[i] > 0]
    weight_list.sort(key=lambda x: x[1])
    
    print(f"  Lowest weights (most common):")
    for name, w in weight_list[:5]:
        print(f"    {name}: {w:.4f}")
    print(f"  Highest weights (least common):")
    for name, w in weight_list[-5:]:
        print(f"    {name}: {w:.4f}")
    
    # Print boosted entities
    if boosted_entities:
        print(f"\n  High-importance entities (boosted by {importance_boost}x):")
        for name, old_w, new_w in boosted_entities:
            print(f"    {name}: {old_w:.4f} -> {new_w:.4f}")

    # Print penalized entities
    if penalized_entities:
        print(f"\n  Low-importance entities (penalized by {penalty_factor}x):")
        for name, old_w, new_w in penalized_entities:
            print(f"    {name}: {old_w:.4f} -> {new_w:.4f}")
    
    return weights


def build_civ_entity_mapping(csv_path: str, filter_events: List[str] = None,
                              filter_entities: List[str] = None) -> Dict[str, set]:
    """Build a mapping of civilization -> valid entities from training data.
    
    Args:
        csv_path: Path to CSV file
        filter_events: List of event types to EXCLUDE (e.g., ['DESTROY'])
        filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep'])
    
    Returns:
        civ_entity_mapping: Dict mapping civ name -> set of valid entity names
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Filter out unwanted events
    if filter_events:
        df = df[~df['event'].isin(filter_events)]
    
    # Filter out unwanted entities
    if filter_entities:
        df = df[~df['entity'].astype(str).isin(filter_entities)]
    
    # Filter out events at timestamp 0 (outside player control)
    if 'time' in df.columns:
        df = df[df['time'] != 0]
    
    # Build mapping from player_civ to valid entities
    civ_entity_mapping = {}
    
    if 'player_civ' in df.columns:
        grouped = df.groupby('player_civ')['entity'].apply(lambda x: set(x.astype(str).unique()))
        civ_entity_mapping = grouped.to_dict()
    
    # Print summary
    print(f"Built civ-entity mapping for {len(civ_entity_mapping)} civilizations:")
    for civ, entities in sorted(civ_entity_mapping.items()):
        print(f"  {civ}: {len(entities)} unique entities")
    
    return civ_entity_mapping


def create_civ_entity_mask(civ_entity_mapping: Dict[str, set], 
                           entity_vocab: Dict[str, int],
                           civ_vocab: Dict[str, int],
                           device: torch.device) -> torch.Tensor:
    """Create a mask tensor of shape (num_civs, vocab_size_entity).
    
    mask[civ_id, entity_id] = True if civ can build that entity, False otherwise.
    
    Args:
        civ_entity_mapping: Dict mapping civ name -> set of valid entity names
        entity_vocab: Entity vocabulary (token -> id)
        civ_vocab: Civilization vocabulary (civ name -> id)
        device: Device to create tensor on
        
    Returns:
        mask: (num_civs, vocab_size_entity) boolean tensor
    """
    num_civs = len(civ_vocab)
    vocab_size_entity = len(entity_vocab)
    
    # Initialize mask - start with all False
    mask = torch.zeros(num_civs, vocab_size_entity, dtype=torch.bool, device=device)
    
    # Special tokens (PAD=0, UNK=1) should always be allowed
    mask[:, 0] = True  # PAD
    mask[:, 1] = True  # UNK
    
    # Fill in valid entities for each civ
    inv_civ_vocab = {v: k for k, v in civ_vocab.items()}
    
    for civ_id in range(num_civs):
        civ_name = inv_civ_vocab.get(civ_id, '')
        
        if civ_name in civ_entity_mapping:
            valid_entities = civ_entity_mapping[civ_name]
            for entity_name in valid_entities:
                if entity_name in entity_vocab:
                    entity_id = entity_vocab[entity_name]
                    mask[civ_id, entity_id] = True
    
    # Log coverage
    total_entities = vocab_size_entity - 2  # Exclude PAD and UNK
    for civ_id in range(2, num_civs):  # Skip PAD and UNK civs
        civ_name = inv_civ_vocab.get(civ_id, '')
        valid_count = mask[civ_id].sum().item() - 2  # Exclude PAD/UNK
        print(f"  {civ_name}: {valid_count}/{total_entities} entities valid ({100*valid_count/total_entities:.1f}%)")
    
    return mask


def log_example_predictions(model, dataloader, entity_vocab, device, wandb):
    """Log example entity predictions to wandb."""
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    entity_ids = batch['entity_ids'].to(device)
    player_civ = batch['player_civ'].to(device)
    enemy_civ = batch['enemy_civ'].to(device)
    map_id = batch['map_id'].to(device)
    mask = batch.get('mask', None)
    if mask is not None:
        mask = mask.to(device)
    
    # Make tensors contiguous
    entity_input = entity_ids[:, :-1].contiguous()
    input_mask = mask[:, :-1].contiguous() if mask is not None else None
    
    # Generate predictions
    with torch.no_grad():
        entity_logits = model(
            entity_input, player_civ, enemy_civ, map_id, input_mask, predict_next=True
        )
        
        entity_probs = torch.softmax(entity_logits, dim=-1)
        entity_preds = torch.argmax(entity_logits, dim=-1)
    
    # Convert to human-readable format
    entity_id_to_token = {v: k for k, v in entity_vocab.items()}
    
    examples = []
    for i in range(min(5, len(entity_ids))):  # Show first 5 examples
        # Get sequence (filter out padding)
        if mask is not None:
            valid_mask = mask[i].cpu().numpy()
            seq_length = valid_mask.sum()
        else:
            seq_length = len(entity_ids[i])
        
        input_entities = [entity_id_to_token.get(idx.item(), '<UNK>') 
                         for idx in entity_ids[i, :seq_length-1]]
        
        true_next_entity = entity_id_to_token.get(entity_ids[i, seq_length-1].item(), '<UNK>')
        pred_next_entity = entity_id_to_token.get(entity_preds[i].item(), '<UNK>')
        
        examples.append({
            'input_sequence': input_entities,
            'true_next': true_next_entity,
            'pred_next': pred_next_entity,
            'correct': true_next_entity == pred_next_entity
        })
    
    # Create wandb table
    table = wandb.Table(columns=[
        "Example", "Input Build Order", "True Next", "Predicted Next", "Correct"
    ])
    
    for i, ex in enumerate(examples):
        input_str = " → ".join(ex['input_sequence'])
        
        table.add_data(
            i + 1,
            input_str,
            ex['true_next'],
            ex['pred_next'],
            "✓" if ex['correct'] else "✗"
        )
    
    wandb.log({"example_predictions": table})
    model.train()

if __name__ == "__main__":
    main()