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

"""
Train SimpleBuildOrderGenerator from sequences.

Usage:
  python BuildOrderPrediction/SimpleBuildOrderPrediction_train.py

  """

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean', ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, C) or (B, L, C) logits (sequence models typically output B, L, C)
            target: (B,) or (B, L) indices
        """
        if input.ndim > 2:
            # (B, L, C) -> (B*L, C)
            c = input.shape[2]  # C is the last dimension for (B, L, C)
            input = input.reshape(-1, c)
            target = target.reshape(-1)

        log_pt = F.log_softmax(input, dim=-1)
        pt = torch.exp(log_pt)
        
        # Calculate cross entropy (without reduction yet)
        ce_loss = F.nll_loss(log_pt, target, reduction='none', ignore_index=self.ignore_index)
        
        # Calculate modulating factor: (1 - pt)^gamma
        pt_gather = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        modulating_factor = (1 - pt_gather) ** self.gamma
        
        # Apply alpha weighting if provided
        weight = None
        if self.alpha is not None:
            self.alpha = self.alpha.to(input.device)
            # Alpha for the target class
            alpha_t = self.alpha[target]
            # Handle ignore_index in alpha lookup (though nll_loss handles it for loss)
            # We can just set alpha to 1 for ignored indices to avoid gather errors if target has ignore_index
            # But standard gather will fail if index is out of bounds (negative).
            # Usually ignore_index is -100.
            
            # To be safe, we rely on ce_loss being 0 for ignore_index, 
            # so the value of alpha_t doesn't matter there, but we must not error on lookup.
             # Create a mask for valid targets
            valid_mask = target != self.ignore_index
            
            # Create a safe target tensor for gathering (replace ignore_index with 0)
            safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
            
            alpha_t = self.alpha[safe_target]
            
            focal_loss = alpha_t * modulating_factor * ce_loss
        else:
            focal_loss = modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            # Compute mean only over non-ignored indices
            valid_count = (target != self.ignore_index).float().sum()
            return focal_loss.sum() / torch.clamp(valid_count, min=1.0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SequencePredictor(nn.Module):
    """
    Causal Transformer for entity build order prediction.
    Uses decoder-style causal masking to prevent looking at future tokens.
    Conditioned on player civilization, enemy civilization, and map.
    
    Key improvements over encoder-only:
    1. Causal attention mask - each position can only attend to past positions
    2. Learnable position embeddings for better pattern learning
    3. Pre-norm architecture for training stability
    4. Embedding scaling for better gradient flow
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
        self.vocab_size_entity = vocab_size_entity
        
        # 1. Entity embedding with scaling
        self.entity_embed = nn.Embedding(vocab_size_entity, d_model)
        self.embed_scale = math.sqrt(d_model)
        
        # 2. Condition embeddings (player civ, enemy civ, map)
        # Use separate embeddings for player vs enemy civ for clarity
        self.player_civ_embed = nn.Embedding(civ_vocab_size, d_model)
        self.enemy_civ_embed = nn.Embedding(civ_vocab_size, d_model)
        self.map_embed = nn.Embedding(map_vocab_size, d_model)
        
        # 3. Learnable position embeddings (better than sinusoidal for this task)
        # +3 for condition tokens, +1 for safety margin
        self.pos_embed = nn.Embedding(max_seq_len + 4, d_model)
        
        # 4. Input layer norm and dropout
        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 5. Transformer decoder layers with causal masking
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        
        # 6. Output layer norm (important for pre-norm architecture)
        self.output_ln = nn.LayerNorm(d_model)
        
        # 7. Output head - simpler is better, avoid over-parameterization
        self.entity_head = nn.Linear(d_model, vocab_size_entity)
        
        # 8. Cache for causal mask
        self._causal_mask_cache = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        # Entity embeddings - small init
        nn.init.normal_(self.entity_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.player_civ_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.enemy_civ_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.map_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        
        # Output head - very small init for residual-like behavior
        nn.init.normal_(self.entity_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.entity_head.bias)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask.
        Returns a mask where True means "do not attend" (masked out).
        
        For sequence [Cond1, Cond2, Cond3, E1, E2, E3]:
        - Condition tokens can attend to each other (fully connected)
        - Entity tokens can attend to all conditions AND previous entities
        """
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            # Create causal mask for the full sequence (including 3 condition tokens)
            # Shape: (seq_len, seq_len)
            # mask[i, j] = True means position i cannot attend to position j
            
            total_len = seq_len + 3  # +3 for condition tokens
            mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)
            
            # Condition tokens (0, 1, 2) can attend to each other
            mask[:3, :3] = False
            
            # Entity tokens can attend to:
            # - All condition tokens
            # - All previous entity tokens (causal)
            for i in range(3, total_len):
                # Can attend to all conditions
                mask[i, :3] = False
                # Can attend to all previous entities (including self)
                mask[i, 3:i+1] = False
            
            self._causal_mask_cache[cache_key] = mask
        
        return self._causal_mask_cache[cache_key]
    
    def forward(self,
                entity_sequence: torch.Tensor,
                player_civ: torch.Tensor,
                enemy_civ: torch.Tensor,
                map_id: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                predict_next: bool = True) -> torch.Tensor:
        """
        Forward pass with causal attention for next-token prediction.
        
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
        
        # 1. Create condition embeddings
        p_civ = self.player_civ_embed(player_civ)   # (B, d_model)
        e_civ = self.enemy_civ_embed(enemy_civ)     # (B, d_model)
        map_emb = self.map_embed(map_id)            # (B, d_model)
        condition = torch.stack([p_civ, e_civ, map_emb], dim=1)  # (B, 3, d_model)
        
        # 2. Create entity embeddings with scaling
        entity_emb = self.entity_embed(entity_sequence) * self.embed_scale  # (B, L, d_model)
        
        # 3. Concatenate: [PlayerCiv, EnemyCiv, Map, Entity1, Entity2, ...]
        combined_emb = torch.cat([condition, entity_emb], dim=1)  # (B, 3+L, d_model)
        
        # 4. Add positional embeddings
        positions = torch.arange(combined_emb.size(1), device=device)
        pos_emb = self.pos_embed(positions)  # (3+L, d_model)
        combined_emb = combined_emb + pos_emb.unsqueeze(0)
        
        # 5. Apply layer norm and dropout
        combined_emb = self.embed_ln(combined_emb)
        combined_emb = self.embed_dropout(combined_emb)
        
        # 6. Create causal attention mask
        causal_mask = self._get_causal_mask(seq_len, device)  # (3+L, 3+L)
        
        # 7. Create padding mask
        if attention_mask is None:
            attention_mask = (entity_sequence != 0)  # (B, L)
        
        # Prepend True for condition tokens
        condition_mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        full_mask = torch.cat([condition_mask, attention_mask], dim=1)  # (B, 3+L)
        src_key_padding_mask = ~full_mask  # True = masked out
        
        # 8. Apply transformer with causal mask
        encoded = self.transformer(
            combined_emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, 3+L, d_model)
        
        # 9. Apply output layer norm
        encoded = self.output_ln(encoded)
        
        # 10. Remove condition tokens
        encoded = encoded[:, 3:, :]  # (B, L, d_model)
        
        if predict_next:
            # Get the last valid position for each sequence
            original_mask = full_mask[:, 3:]
            last_valid_indices = original_mask.sum(dim=1) - 1  # (B,)
            
            # Gather hidden states at last valid positions
            batch_indices = torch.arange(batch_size, device=device)
            last_hidden = encoded[batch_indices, last_valid_indices]  # (B, d_model)
            
            # Predict next entity
            entity_logits = self.entity_head(last_hidden)  # (B, vocab_size_entity)
        else:
            # Predict for all positions (teacher forcing)
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
        # batch_size = entity_sequence.shape[0]  # Unused
        
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
                 label_smoothing: float = 0.0,
                 use_focal_loss: bool = True,
                 focal_loss_gamma: float = 2.0):
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
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        
        # Initialize Focal Loss if enabled
        if self.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=self.entity_class_weights,
                gamma=self.focal_loss_gamma,
                reduction='mean',
                ignore_index=-100 # Standard ignore index
            )
        
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
        Compute loss for entity sequence prediction with masked label smoothing or Focal Loss.
        
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
        if self.use_focal_loss:
            # Use Focal Loss
            # Focal Loss expects same shape as cross_entropy input
            if entity_logits.dim() == 3:
                 # (B, L, V) -> (B, V, L) if using cross_entropy usually, but our FocalLoss handles (B, C, L) check
                 pass # Our focal loss handles dimensions
            
            entity_loss = self.criterion(entity_logits, entity_targets)
            
        elif self.label_smoothing > 0:
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
        total_correct_top5 = 0
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
                
                # Compute Top-5 Accuracy
                _, top5_preds = entity_logits.topk(5, dim=-1)  # (B, 5)
                # Check if target is in top 5
                # Expand targets to (B, 1) to compare
                top5_correct = top5_preds.eq(entity_targets.view(-1, 1).expand_as(top5_preds))
                total_correct_top5 += top5_correct.sum().item()
                
                total_loss += loss.item() * entity_ids.size(0)
                total_correct_entity += (entity_preds == entity_targets).sum().item()
                total_samples += entity_ids.size(0)
        
        avg_loss = total_loss / total_samples
        entity_accuracy = total_correct_entity / total_samples
        entity_top5_accuracy = total_correct_top5 / total_samples
        
        return {
            'loss': avg_loss,
            'entity_accuracy': entity_accuracy,
            'entity_top5_accuracy': entity_top5_accuracy
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
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (smaller for better generalization)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (higher with warmup)')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length (shorter = more focused on early game)')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout rate (slightly higher for regularization)')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                       help='Teacher forcing ratio (1.0 = always use teacher forcing for stable training)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                       help='Label smoothing factor (lower to not hurt accuracy)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs for learning rate')
    
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
        weight_decay=0.05,  # Stronger regularization
        betas=(0.9, 0.98),  # Standard transformer betas
        eps=1e-8
    )
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum_steps
    
    # Create learning rate scheduler with warmup and cosine decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Min LR = 10% of max
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
            'architecture': 'CausalSequencePredictor',
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
            'map_vocab_size': len(map_vocab),
            'warmup_epochs': args.warmup_epochs,
            'label_smoothing': args.label_smoothing,
            'weight_decay': 0.05,
            'total_params': sum(p.numel() for p in model.parameters()),
            'causal_attention': True,  # Key improvement
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
    patience = 15  # Early stopping patience (increased for longer training)
    global_step = 0
    
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
            
            # Step scheduler after each optimizer step
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scheduler.step()
                global_step += 1
            
            # Log step-level training loss to W&B
            if not args.no_wandb:
                wandb.log({
                    'step': global_step,
                    'train_step/loss': loss_dict['loss'],
                    'train_step/lr': optimizer.param_groups[0]['lr'],
                }, step=global_step)
            
            # Log batch progress
            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss_dict['loss']:.4f}, LR: {current_lr:.6f}")
        
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
        print(f"  Top-5 Accuracy: {val_metrics['entity_top5_accuracy']:.4f}")
        
        # Get current learning rate
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
                'val/entity_top5_accuracy': val_metrics['entity_top5_accuracy'],
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
                                  filter_entities: List[str] = None) -> torch.Tensor:
    """Compute tiered class weights for entity loss with strategic importance.
    
    This function uses a tiered weighting approach that balances:
    1. Routine entities (Villager, House, Farm, Wall) - need high accuracy, moderate weight
    2. Strategic timing-critical entities (Age-ups, early military) - boosted weight
    3. Standard entities - use log-dampened inverse frequency
    4. Very rare entities - capped to prevent over-emphasis on noise
    
    The approach avoids pure inverse frequency which would over-penalize common 
    entities that we MUST predict correctly (Villagers are 24% of all events).
    
    Args:
        csv_path: Path to CSV file
        entity_vocab: Entity vocabulary (token -> id)
        filter_events: List of event types to EXCLUDE
        filter_entities: List of entity names to EXCLUDE (e.g., ['Sheep'])
        
    Returns:
        weights: (vocab_size,) tensor of class weights
    """
    import pandas as pd
    
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
    total_count = entity_counts.sum()
    
    # === Define Entity Categories ===
    
    # Tier 1: High-frequency routine entities - MUST predict correctly
    # These are the backbone of every game, penalizing them too much hurts accuracy
    routine_entities = {
        'Villager': 1.0,        # Most common - base weight
        'Gilded Villager': 1.0,
        'Wheat Field': 1.0,     # Farms
        'Palisade Wall': 0.8,   # Walls are common but less strategic
        'Stone Wall': 0.8,
        'House': 0.6,           # Houses are predictable, less weight needed
        'Lumber Camp': 1.2,     # Resource buildings matter for build order
        'Gold Mining Camp': 1.2,
        'Gristmill': 1.2,
        'Farmhouse': 1.2,
    }
    
    # Tier 2: Strategic timing-critical entities - boost significantly
    # These mark key strategic decisions and transitions
    strategic_entities = {
        # Age-ups are THE most critical timing decisions
        'Age Display Persistent 2': 8.0,  # Feudal timing
        'Age Display Persistent 3': 10.0, # Castle timing - game-defining
        'Age Display Persistent 4': 12.0, # Imperial timing
        
        # Military production buildings - timing of first building matters
        'Barracks': 3.0,
        'Archery Range': 3.0,
        'Stable': 4.0,          # Stable timing often signals knight play
        'Siege Workshop': 5.0,  # Late game transition
        
        # Economic/Tech buildings with strategic timing
        'Blacksmith': 3.0,      # Upgrade timing matters
        'University': 4.0,      # Castle age tech
        'Town Center': 5.0,     # TC timing is strategic (boom vs aggression)
        'Keep': 4.0,
        'Castle': 5.0,
        
        # Early-age military units (rare but timing-critical)
        'Knight 2': 6.0,        # Feudal knights are rare and strategic
        'Manatarms 2': 5.0,     # Early MAA rush
        'Crossbowman 2': 4.0,   # Early crossbow
        'Lancer 2': 5.0,        # Early lancer
        
        # Age 3/4 power units (less rare but still strategic)
        'Knight 3': 2.5,
        'Knight 4': 3.0,
        'Manatarms 3': 2.0,
        'Manatarms 4': 2.5,
    }
    
    # Tier 3: Low-priority entities - reduce weight
    low_priority_entities = {
        'Outpost': 0.5,  # Common but not strategic
    }
    
    # === Compute Base Weights Using Log-Dampened Inverse Frequency ===
    # log-dampening prevents extreme weights for rare entities
    # Formula: weight = log(total_count / count) / log(total_count / median_count)
    
    vocab_size = len(entity_vocab)
    weights = torch.ones(vocab_size)
    
    median_count = entity_counts.median()
    
    for entity_name, entity_id in entity_vocab.items():
        if entity_name in ('<PAD>', '<UNK>', '<BOS>'):
            weights[entity_id] = 0.0  # Ignore special tokens
        elif entity_name in entity_counts.index:
            count = entity_counts[entity_name]
            
            # Log-dampened inverse frequency (smoother than pure inverse)
            # This gives more balanced weights across the frequency spectrum
            log_weight = np.log1p(total_count / count) / np.log1p(total_count / median_count)
            weights[entity_id] = log_weight
        else:
            weights[entity_id] = 1.0  # Default weight for unseen entities
    
    # === Apply Tiered Category Multipliers ===
    
    applied_weights = {'routine': [], 'strategic': [], 'low_priority': []}
    
    for entity_name, multiplier in routine_entities.items():
        if entity_name in entity_vocab:
            entity_id = entity_vocab[entity_name]
            old_weight = weights[entity_id].item()
            weights[entity_id] = weights[entity_id] * multiplier
            applied_weights['routine'].append((entity_name, old_weight, weights[entity_id].item()))
    
    for entity_name, multiplier in strategic_entities.items():
        if entity_name in entity_vocab:
            entity_id = entity_vocab[entity_name]
            old_weight = weights[entity_id].item()
            weights[entity_id] = weights[entity_id] * multiplier
            applied_weights['strategic'].append((entity_name, old_weight, weights[entity_id].item()))
    
    for entity_name, multiplier in low_priority_entities.items():
        if entity_name in entity_vocab:
            entity_id = entity_vocab[entity_name]
            old_weight = weights[entity_id].item()
            weights[entity_id] = weights[entity_id] * multiplier
            applied_weights['low_priority'].append((entity_name, old_weight, weights[entity_id].item()))
    
    # === Cap Extreme Weights ===
    # Very rare entities shouldn't dominate the loss
    valid_mask = weights > 0
    if valid_mask.sum() > 0:
        # Cap at 99th percentile to prevent outlier domination
        cap_value = torch.quantile(weights[valid_mask], 0.99).item()
        weights = torch.clamp(weights, max=cap_value * 1.5)  # Allow strategic boost above cap
    
    # === Normalize to Mean = 1.0 ===
    valid_mask = weights > 0
    if valid_mask.sum() > 0:
        mean_weight = weights[valid_mask].mean()
        weights[valid_mask] = weights[valid_mask] / mean_weight
    
    # === Print Detailed Summary ===
    print(f"\n{'='*60}")
    print(f"Entity Class Weights Summary (Tiered Strategic Weighting)")
    print(f"{'='*60}")
    print(f"  Total entities: {len(entity_counts)}")
    print(f"  Min weight: {weights[valid_mask].min():.4f}")
    print(f"  Max weight: {weights[valid_mask].max():.4f}")
    print(f"  Mean weight: {weights[valid_mask].mean():.4f}")
    print(f"  Median weight: {torch.median(weights[valid_mask]):.4f}")
    
    # Show weight distribution by category
    inv_vocab = {v: k for k, v in entity_vocab.items()}
    weight_list = [(inv_vocab[i], weights[i].item()) for i in range(vocab_size) if weights[i] > 0]
    weight_list.sort(key=lambda x: x[1])
    
    print(f"\n  Lowest weights (high-frequency entities):")
    for name, w in weight_list[:8]:
        count = entity_counts.get(name, 0)
        print(f"    {name}: {w:.4f} (count: {count:,})")
    
    print(f"\n  Highest weights (strategic/rare entities):")
    for name, w in weight_list[-8:]:
        count = entity_counts.get(name, 0)
        print(f"    {name}: {w:.4f} (count: {count:,})")
    
    # Print category summaries
    if applied_weights['routine']:
        print(f"\n  Routine entities (accuracy-critical):")
        for name, old_w, new_w in sorted(applied_weights['routine'], key=lambda x: -x[2])[:5]:
            print(f"    {name}: {new_w:.4f}")
    
    if applied_weights['strategic']:
        print(f"\n  Strategic entities (timing-critical, boosted):")
        for name, old_w, new_w in sorted(applied_weights['strategic'], key=lambda x: -x[2])[:10]:
            print(f"    {name}: {old_w:.4f} -> {new_w:.4f}")
    
    # Show key ratios
    villager_weight = weights[entity_vocab.get('Villager', 0)].item() if 'Villager' in entity_vocab else 0
    knight2_weight = weights[entity_vocab.get('Knight 2', 0)].item() if 'Knight 2' in entity_vocab else 0
    age3_weight = weights[entity_vocab.get('Age Display Persistent 3', 0)].item() if 'Age Display Persistent 3' in entity_vocab else 0
    
    print(f"\n  Key Weight Ratios:")
    print(f"    Villager: {villager_weight:.4f}")
    print(f"    Knight 2 / Villager: {knight2_weight / max(villager_weight, 0.001):.2f}x")
    print(f"    Age-up Castle / Villager: {age3_weight / max(villager_weight, 0.001):.2f}x")
    print(f"{'='*60}\n")
    
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