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
  python BuildOrderPrediction/MoE_WithDecoder_train.py
  python BuildOrderPrediction/MoE_WithDecoder_train.py --wins_only --max_seq_len 50 --csv_path training_data_2026_01.csv 


  Last run parameters:
python BuildOrderPrediction/MoE_WithDecoder_train.py \
  --wins_only --max_seq_len 50 --csv_path training_data_2026_01.csv \
  --d_model 1024 --nhead 16 --num_decoder_layers 12 --dim_feedforward 4096 \
  --batch_size 80 --grad_accum_steps 3 --dropout 0.1 \
  --num_experts 8 --use_moe --use_ngram --use_rope --use_contrastive \
  --epochs 70 --pos_weight_max 5.0 --late_focus_prob 0.4
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

        # Clamp logits for numerical stability before softmax
        # This prevents -inf values (from civ masking) creating NaN in fp16
        input = input.float()  # Always compute loss in fp32
        input = torch.clamp(input, min=-1e4)
        
        log_pt = F.log_softmax(input, dim=-1)
        pt = torch.exp(log_pt)
        
        # Calculate cross entropy (without reduction yet)
        ce_loss = F.nll_loss(log_pt, target, reduction='none', ignore_index=self.ignore_index)
        
        # Calculate modulating factor: (1 - pt)^gamma
        # Use safe target for gathering (replace ignore_index with 0 to avoid OOB)
        safe_target_for_pt = torch.where(target != self.ignore_index, target, torch.zeros_like(target))
        pt_gather = pt.gather(1, safe_target_for_pt.unsqueeze(1)).squeeze(1)
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


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better relative position modeling.
    This helps the model learn position-invariant patterns in build orders.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions."""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)  # (seq_len, d_model/2)
        
        # Create cos and sin embeddings
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        seq_len = x.shape[1]
        
        # Get cos/sin for this sequence length
        cos = self.cos_cached[offset:offset + seq_len]  # (seq_len, d_model)
        sin = self.sin_cached[offset:offset + seq_len]  # (seq_len, d_model)
        
        # Apply rotation
        x_rot = self._rotate_half(x)
        return x * cos.unsqueeze(0) + x_rot * sin.unsqueeze(0)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention for better condition fusion.
    Uses a learnable gate to control how much condition information flows in.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply gated cross-attention.
        Args:
            x: (B, L, d_model) entity representations
            condition: (B, C, d_model) condition tokens
        Returns:
            x_updated: (B, L, d_model)
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(x, condition, condition)
        attn_out = self.dropout(attn_out)
        
        # Compute gate from concatenation of x and attention output
        gate_input = torch.cat([x, attn_out], dim=-1)
        gate = self.gate(gate_input)
        
        # Apply gated residual
        x = self.norm(x + gate * attn_out)
        return x


class LocalAttentionBlock(nn.Module):
    """
    Local attention block that focuses on nearby tokens.
    Useful for capturing local build order patterns (e.g., Villager -> Farm sequences).
    """
    def __init__(self, d_model: int, nhead: int, window_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply local attention with window masking."""
        B, L, D = x.shape
        device = x.device
        
        # Create local attention mask (causal within window)
        local_mask = torch.ones(L, L, dtype=torch.bool, device=device)
        for i in range(L):
            start = max(0, i - self.window_size + 1)
            local_mask[i, start:i+1] = False  # Attend to window (causal)
        
        # Apply attention
        attn_out, _ = self.attn(x, x, x, attn_mask=local_mask)
        x = self.norm(x + self.dropout(attn_out))
        return x


class NGramFeatureExtractor(nn.Module):
    """
    Extract n-gram features from entity sequences.
    Captures common patterns like "Villager, Villager, House" or "Barracks, Spearman".
    """
    def __init__(self, d_model: int, max_ngram: int = 4):
        super().__init__()
        self.max_ngram = max_ngram
        
        # Convolutions for different n-gram sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // max_ngram, kernel_size=n, padding=n-1)
            for n in range(1, max_ngram + 1)
        ])
        
        # Combine n-gram features
        self.combine = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract n-gram features.
        Args:
            x: (B, L, d_model)
        Returns:
            features: (B, L, d_model)
        """
        # Transpose for conv1d: (B, d_model, L)
        x_t = x.transpose(1, 2)
        
        # Apply each n-gram convolution
        ngram_features = []
        for conv in self.convs:
            feat = conv(x_t)
            # Truncate to original length (remove padding effects)
            feat = feat[:, :, :x.shape[1]]
            ngram_features.append(feat)
        
        # Concatenate and transpose back
        combined = torch.cat(ngram_features, dim=1).transpose(1, 2)  # (B, L, d_model)
        
        # Project and normalize
        return self.norm(x + self.combine(combined))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer for specialized processing of different entity types.
    Different experts can specialize in: economy, military, tech, timing, etc.
    """
    def __init__(self, d_model: int, num_experts: int = 4, expert_dim: int = None, 
                 dropout: float = 0.1, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        if expert_dim is None:
            expert_dim = d_model * 4
        
        # Router network
        self.router = nn.Linear(d_model, num_experts)
        
        # Expert networks (each is a small FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Load balancing loss coefficient
        self.aux_loss_coef = 0.01
        self.aux_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mixture of experts.
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        B, L, D = x.shape
        
        # Compute router logits and get top-k experts
        router_logits = self.router(x)  # (B, L, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Compute auxiliary load balancing loss
        # Encourage even distribution across experts
        avg_probs = router_probs.mean(dim=[0, 1])  # (num_experts,)
        target_prob = 1.0 / self.num_experts
        self.aux_loss = self.aux_loss_coef * ((avg_probs - target_prob) ** 2).sum()
        
        # Apply experts (using loop for clarity, can be optimized)
        expert_outputs = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (top_k_indices[:, :, k] == e)  # (B, L)
                if mask.any():
                    expert_input = x[mask]  # (N, D)
                    expert_out = self.experts[e](expert_input)
                    weight = top_k_probs[:, :, k][mask].unsqueeze(-1)  # (N, 1)
                    expert_outputs[mask] = expert_outputs[mask] + weight * expert_out
        
        return self.norm(x + expert_outputs)
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load balancing loss."""
        return self.aux_loss


class ImprovedTransformerBlock(nn.Module):
    """
    Improved transformer block with:
    - Pre-norm architecture
    - Gated residual connections
    - Optional local attention
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float = 0.1, use_local_attn: bool = False):
        super().__init__()
        
        # Self-attention with pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Optional local attention
        self.use_local_attn = use_local_attn
        if use_local_attn:
            self.local_attn = LocalAttentionBlock(d_model, nhead, window_size=8, dropout=dropout)
        
        # FFN with pre-norm
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Gated residual connections
        self.gate1 = nn.Parameter(torch.ones(1) * 0.5)
        self.gate2 = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply transformer block with gated residuals."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, 
                                      attn_mask=attn_mask, 
                                      key_padding_mask=key_padding_mask)
        x = x + self.gate1 * self.dropout1(attn_out)
        
        # Optional local attention
        if self.use_local_attn:
            x = self.local_attn(x)
        
        # FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.gate2 * ffn_out
        
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Enhanced transformer decoder block with:
    - Pre-norm architecture
    - Causal self-attention
    - Cross-attention to condition memory (NOT an encoder - just MLP-projected conditions)
    - Gated residual connections
    - Optional local attention
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float = 0.1, use_local_attn: bool = False):
        super().__init__()
        
        # Self-attention with pre-norm (causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention to condition memory
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional local attention
        self.use_local_attn = use_local_attn
        if use_local_attn:
            self.local_attn = LocalAttentionBlock(d_model, nhead, window_size=8, dropout=dropout)
        
        # FFN with pre-norm
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Gated residual connections
        self.gate1 = nn.Parameter(torch.ones(1) * 0.5)
        self.gate2 = nn.Parameter(torch.ones(1) * 0.5)
        self.gate3 = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor, 
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply decoder block with cross-attention to condition memory.
        
        Args:
            x: (B, L_tgt, d_model) decoder input
            memory: (B, L_mem, d_model) condition memory (MLP-projected conditions)
            tgt_mask: causal mask for self-attention
            tgt_key_padding_mask: padding mask for decoder input
            memory_key_padding_mask: padding mask for condition memory
        """
        # Causal self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, 
                                      attn_mask=tgt_mask, 
                                      key_padding_mask=tgt_key_padding_mask)
        x = x + self.gate1 * self.dropout1(attn_out)
        
        # Cross-attention to condition memory
        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm, memory, memory,
                                        key_padding_mask=memory_key_padding_mask)
        x = x + self.gate2 * self.dropout2(cross_out)
        
        # Optional local attention
        if self.use_local_attn:
            x = self.local_attn(x)
        
        # FFN
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.gate3 * ffn_out
        
        return x


class SequencePredictor(nn.Module):
    """
    Conditional Decoder Transformer for entity build order prediction.
    
    Architecture:
    - Condition Projection: MLP that combines (player_civ, enemy_civ, map) into a single memory vector
    - Decoder: Autoregressively generates entity sequence with cross-attention to condition memory
    
    NOTE: This is NOT an encoder-decoder architecture. The "condition projection" is just an MLP,
    not a transformer encoder. Self-attention on a single token (the condition vector) would be
    a no-op, so we removed the useless encoder transformer layers. The decoder cross-attends to
    the projected condition vector at every layer.
    
    Key improvements over basic transformer:
    1. Rotary Position Embeddings (RoPE) for better relative position modeling
    2. Gated cross-attention for condition fusion at every decoder layer
    3. N-gram feature extraction for local patterns
    4. Mixture of Experts for specialized processing
    5. Multi-scale attention (local + global) in decoder
    6. Contrastive representation learning auxiliary objective
    7. Gated residual connections for training stability
    8. Deep output head with skip connection
    9. Per-layer map cross-attention for persistent map influence
    """
    
    def __init__(self,
                 vocab_size_entity: int,
                 civ_vocab_size: int,
                 map_vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.15,
                 max_seq_len: int = 512,
                 num_experts: int = 4,
                 use_moe: bool = True,
                 use_ngram: bool = True,
                 use_rope: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size_entity = vocab_size_entity
        self.use_moe = use_moe
        self.use_ngram = use_ngram
        self.use_rope = use_rope
        self.num_decoder_layers = num_decoder_layers
        
        # 1. Entity embedding with scaling
        self.entity_embed = nn.Embedding(vocab_size_entity, d_model)
        self.embed_scale = math.sqrt(d_model)
        
        # 2. Condition embeddings (player civ, enemy civ, map)
        self.player_civ_embed = nn.Embedding(civ_vocab_size, d_model)
        self.enemy_civ_embed = nn.Embedding(civ_vocab_size, d_model)
        self.map_embed = nn.Embedding(map_vocab_size, d_model)
        
        # Condition projection MLP (combine all conditions into a single memory vector)
        # NOTE: This replaces what was previously called an "encoder". Self-attention on a
        # single token is a no-op, so we just use an MLP to project the concatenated conditions.
        # The decoder cross-attends to this projected condition vector at every layer.
        self.condition_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)  # Extra layer for more expressive condition encoding
        )
        
        # 3. Position embeddings (with optional RoPE)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(d_model, max_seq_len + 4)
        else:
            self.pos_embed = nn.Embedding(max_seq_len + 4, d_model)
        
        # 4. N-gram feature extractor for local patterns
        if use_ngram:
            self.ngram_extractor = NGramFeatureExtractor(d_model, max_ngram=4)
        
        # 5. Gated cross-attention for condition fusion
        self.condition_cross_attn = GatedCrossAttention(d_model, nhead, dropout)
        
        # 6. Input layer norm and dropout
        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 7. Condition output layer norm (applied after condition projection)
        self.condition_output_ln = nn.LayerNorm(d_model)
        
        # 8. DECODER: Transformer decoder layers with cross-attention to condition memory
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_local_attn=(i % 2 == 0)  # Alternate local attention
            )
            for i in range(num_decoder_layers)
        ])
        
        # 9. Decoder Mixture of Experts (applied every other layer)
        if use_moe:
            self.decoder_moe_layers = nn.ModuleList([
                MixtureOfExperts(d_model, num_experts=num_experts, dropout=dropout)
                if i % 2 == 1 else None
                for i in range(num_decoder_layers)
            ])
        
        # 10. Map-specific cross-attention for each decoder layer
        # This ensures map information is explicitly used at every decoder layer
        self.map_cross_attn_layers = nn.ModuleList([
            GatedCrossAttention(d_model, nhead, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Map influence gate (learnable weight for map contribution)
        self.map_influence_gate = nn.Parameter(torch.tensor(0.2))  # Initial 20% influence
        
        # 11. Decoder output layer norm
        self.decoder_output_ln = nn.LayerNorm(d_model)
        
        # 12. Deep output head with skip connection and bottleneck
        self.entity_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.entity_classifier = nn.Linear(d_model, vocab_size_entity)
        
        # Skip connection gate for output head
        self.output_gate = nn.Parameter(torch.tensor(0.5))
        
        # 13. Contrastive projection head (for auxiliary loss)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2)
        )
        
        # 14. Cache for causal mask
        self._causal_mask_cache = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled initialization for deep networks."""
        # Scale based on number of decoder layers
        layer_scale = (2 * self.num_decoder_layers) ** -0.5
        
        # Entity embeddings
        nn.init.normal_(self.entity_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.player_civ_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.enemy_civ_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.map_embed.weight, mean=0.0, std=0.02)
        
        if not self.use_rope:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        
        # Output head - scaled initialization
        for module in self.entity_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * layer_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.normal_(self.entity_classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.entity_classifier.bias)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask.
        Returns a mask where True means "do not attend" (masked out).
        """
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            # Simple causal mask - each position attends only to previous positions
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[cache_key] = mask
        
        return self._causal_mask_cache[cache_key]
    
    def get_moe_aux_loss(self) -> torch.Tensor:
        """Get combined auxiliary loss from all MoE layers in the decoder."""
        if not self.use_moe:
            return torch.tensor(0.0)
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        # Decoder MoE losses only (no encoder MoE since we removed the useless encoder layers)
        for moe in self.decoder_moe_layers:
            if moe is not None:
                total_loss = total_loss + moe.get_aux_loss()
        return total_loss
    
    def compute_condition_memory(self,
                                   player_civ: torch.Tensor,
                                   enemy_civ: torch.Tensor,
                                   map_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project conditions (player civ, enemy civ, map) into memory for decoder cross-attention.
        
        NOTE: This is NOT an encoder! Self-attention on a single token is a no-op.
        We simply project the concatenated condition embeddings through an MLP.
        The decoder cross-attends to this condition memory at every layer.
        
        Args:
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            map_id: (B,) map IDs
            
        Returns:
            condition_memory: (B, 1, d_model) projected condition memory for cross-attention
            map_emb: (B, 1, d_model) separate map embedding for per-layer map cross-attention
        """
        # Create condition embeddings
        p_civ = self.player_civ_embed(player_civ)   # (B, d_model)
        e_civ = self.enemy_civ_embed(enemy_civ)     # (B, d_model)
        map_emb = self.map_embed(map_id)            # (B, d_model)
        
        # Combine conditions via MLP projection (NOT a transformer encoder!)
        condition_concat = torch.cat([p_civ, e_civ, map_emb], dim=-1)  # (B, 3*d_model)
        condition = self.condition_proj(condition_concat)  # (B, d_model)
        
        # Ensure we have a sequence dimension for cross-attention
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)  # (B, 1, d_model)
        
        # Apply layer norm to the projected condition
        condition_memory = self.condition_output_ln(condition)  # (B, 1, d_model)
        
        # Prepare map embedding for decoder (ensure it has sequence dimension)
        if map_emb.dim() == 2:
            map_emb = map_emb.unsqueeze(1)  # (B, 1, d_model)
        
        return condition_memory, map_emb
    
    def forward(self,
                entity_sequence: torch.Tensor,
                player_civ: torch.Tensor,
                enemy_civ: torch.Tensor,
                map_id: torch.Tensor,
                condition_memory: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                predict_next: bool = True,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass with conditional decoder architecture.
        
        Args:
            entity_sequence: (B, L) entity token ids (padded) - decoder input
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            map_id: (B,) map IDs
            condition_memory: (B, 1, d_model) optional pre-computed condition memory
            attention_mask: (B, L) boolean mask (1 for valid positions)
            predict_next: if True, predict next element; if False, predict all elements
            return_embeddings: if True, also return embeddings for contrastive loss
            
        Returns:
            entity_logits: (B, vocab_size_entity) or (B, L, vocab_size_entity)
            [optional] embeddings: (B, d_model // 2) for contrastive learning
        """
        batch_size, seq_len = entity_sequence.shape
        device = entity_sequence.device
        
        # 1. Compute condition memory if not provided
        if condition_memory is None:
            condition_memory, map_emb_memory = self.compute_condition_memory(player_civ, enemy_civ, map_id)
        else:
            # If condition_memory is pre-computed, we still need map embedding for decoder
            map_emb_memory = self.map_embed(map_id)  # (B, d_model)
            if map_emb_memory.dim() == 2:
                map_emb_memory = map_emb_memory.unsqueeze(1)  # (B, 1, d_model)
        
        # 2. Create entity embeddings with scaling (decoder input)
        entity_emb = self.entity_embed(entity_sequence) * self.embed_scale  # (B, L, d_model)
        
        # 3. Apply N-gram feature extraction
        if self.use_ngram:
            entity_emb = self.ngram_extractor(entity_emb)
        
        # 4. Apply position embeddings
        if self.use_rope:
            entity_emb = self.rope(entity_emb)
        else:
            positions = torch.arange(seq_len, device=device)
            pos_emb = self.pos_embed(positions)
            entity_emb = entity_emb + pos_emb.unsqueeze(0)
        
        # 5. Apply gated cross-attention for initial condition fusion (before decoder)
        entity_emb = self.condition_cross_attn(entity_emb, condition_memory)
        
        # 6. Apply layer norm and dropout
        x = self.embed_ln(entity_emb)
        x = self.embed_dropout(x)
        
        # 7. Create masks
        causal_mask = self._get_causal_mask(seq_len, device)
        
        if attention_mask is None:
            attention_mask = (entity_sequence != 0)
        tgt_key_padding_mask = ~attention_mask
        
        # 8. Apply DECODER layers with cross-attention to condition memory
        for i, layer in enumerate(self.decoder_layers):
            x = layer(
                x, 
                memory=condition_memory,
                tgt_mask=causal_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None  # Condition memory has no padding
            )
            
            # Apply MoE if available for this layer
            if self.use_moe and self.decoder_moe_layers[i] is not None:
                x = self.decoder_moe_layers[i](x)
            
            # Apply map-specific cross-attention at each layer
            # This ensures map information influences generation at every step
            # GatedCrossAttention already contains a residual (returns norm(x + gate*attn_out)),
            # so we INTERPOLATE instead of adding, to avoid double-residual magnitude inflation.
            map_context = self.map_cross_attn_layers[i](x, map_emb_memory)
            gate = torch.sigmoid(self.map_influence_gate)  # ensure [0, 1]
            x = x * (1 - gate) + map_context * gate
        
        # 9. Apply decoder output layer norm
        decoded = self.decoder_output_ln(x)
        
        # 10. Get relevant hidden states for prediction
        if predict_next:
            # Get the last valid position for each sequence
            last_valid_indices = attention_mask.sum(dim=1) - 1  # (B,)
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            
            # Gather hidden states at last valid positions
            batch_indices = torch.arange(batch_size, device=device)
            last_hidden = decoded[batch_indices, last_valid_indices]  # (B, d_model)
            
            # Apply deep output head with skip connection
            head_out = self.entity_head(last_hidden)
            gated_hidden = last_hidden * (1 - self.output_gate) + head_out * self.output_gate
            
            # Predict next entity
            entity_logits = self.entity_classifier(gated_hidden)  # (B, vocab_size_entity)
            
            if return_embeddings:
                # Project for contrastive learning
                contrastive_emb = self.contrastive_proj(last_hidden)
                return entity_logits, contrastive_emb
        else:
            # Predict for all positions (teacher forcing)
            head_out = self.entity_head(decoded)
            gated_hidden = decoded * (1 - self.output_gate) + head_out * self.output_gate
            entity_logits = self.entity_classifier(gated_hidden)  # (B, L, vocab_size_entity)
            
            if return_embeddings:
                # Use mean pooling for sequence-level contrastive embedding
                mask_expanded = attention_mask.unsqueeze(-1).float()
                mean_hidden = (decoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                contrastive_emb = self.contrastive_proj(mean_hidden)
                return entity_logits, contrastive_emb
        
        return entity_logits
    
    def generate(self,
                 entity_sequence: torch.Tensor,
                 player_civ: torch.Tensor,
                 enemy_civ: torch.Tensor,
                 map_id: torch.Tensor,
                 max_new_tokens: int = 10,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
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
            top_p: if not None, use nucleus (top-p) sampling
            
        Returns:
            generated_entities: (B, L + max_new_tokens)
        """
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
            
            # Apply nucleus sampling if specified
            if top_p is not None:
                entity_logits = self._top_p_filtering(entity_logits, top_p)
            
            # Sample from distribution
            entity_probs = torch.softmax(entity_logits, dim=-1)
            next_entity = torch.multinomial(entity_probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            current_entities = torch.cat([current_entities, next_entity], dim=1)
        
        return current_entities
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        values, _ = torch.topk(logits, k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits


class SequenceAugmenter:
    """
    Data augmentation for entity sequences.
    Helps the model learn more robust patterns.
    """
    def __init__(self, 
                 swap_prob: float = 0.05,
                 drop_prob: float = 0.05,
                 repeat_prob: float = 0.03):
        self.swap_prob = swap_prob
        self.drop_prob = drop_prob
        self.repeat_prob = repeat_prob
    
    def augment(self, entity_ids: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to entity sequences.
        
        Args:
            entity_ids: (B, L) entity token ids
            mask: (B, L) validity mask
            
        Returns:
            augmented_ids: (B, L) augmented entity ids
            augmented_mask: (B, L) updated mask
        """
        B, L = entity_ids.shape
        augmented_ids = entity_ids.clone()
        augmented_mask = mask.clone()
        
        for b in range(B):
            # Get valid sequence length (excluding BOS and padding)
            valid_len = mask[b].sum().item()
            if valid_len <= 3:  # Too short to augment
                continue
            
            # Random adjacent swaps (preserves local structure mostly)
            if random.random() < self.swap_prob:
                # Pick a random position to swap (not BOS, not last)
                pos = random.randint(2, valid_len - 2)  # Skip BOS at position 0/1
                augmented_ids[b, pos], augmented_ids[b, pos + 1] = \
                    augmented_ids[b, pos + 1].clone(), augmented_ids[b, pos].clone()
            
            # Note: Drop and repeat are more aggressive - use sparingly
            # These can hurt performance if overused, so we keep probs low
        
        return augmented_ids, augmented_mask


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for learning better sequence representations.
    Sequences from the same game/civ should be closer in embedding space.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) labels indicating which samples should be close
                   (e.g., same civ_id or same game_id)
        
        Returns:
            loss: scalar contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)
        
        # Create positive mask (same label = positive pair)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        
        # Remove self-similarity from positive mask
        positive_mask.fill_diagonal_(0)
        
        # If no positive pairs exist, return 0 loss
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute log-softmax for contrastive loss
        # Mask out self-similarity with large negative value
        similarity.fill_diagonal_(float('-inf'))
        
        # For each sample, compute loss against all negatives
        log_prob = F.log_softmax(similarity, dim=1)
        
        # Average log probability over positive pairs
        # CRITICAL: Use masked_fill instead of multiplication to avoid 0 * -inf = NaN.
        # log_prob has -inf on the diagonal (from similarity masking), and
        # positive_mask has 0.0 on the diagonal, so 0.0 * -inf = NaN in IEEE 754.
        masked_log_prob = log_prob.masked_fill(positive_mask == 0, 0.0)
        loss = -masked_log_prob.sum() / positive_mask.sum().clamp(min=1)
        
        return loss


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
    Enhanced trainer for the SequencePredictor model with:
    - Contrastive learning auxiliary loss
    - Sequence augmentation
    - MoE load balancing loss
    - Improved evaluation metrics
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
                 focal_loss_gamma: float = 2.0,
                 use_contrastive: bool = True,
                 contrastive_weight: float = 0.1,
                 use_augmentation: bool = True,
                 augment_prob: float = 0.3,
                 pos_weight_max: float = 3.0,
                 late_focus_prob: float = 0.4):
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
                reduction='none',
                ignore_index=-100
            )
        
        # Contrastive learning
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Sequence augmentation
        self.use_augmentation = use_augmentation
        self.augment_prob = augment_prob
        if use_augmentation:
            self.augmenter = SequenceAugmenter(swap_prob=0.05, drop_prob=0.02, repeat_prob=0.02)
        
        # AMP setup
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        self.grad_accum_steps = grad_accum_steps
        self._accum_step = 0
        self.pos_weight_max = pos_weight_max
        self.late_focus_prob = late_focus_prob
        
    def compute_loss(self,
                entity_logits: torch.Tensor,
                entity_targets: torch.Tensor,
                player_civ: Optional[torch.Tensor] = None,
                pos_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for entity sequence prediction with masked label smoothing or Focal Loss.
        
        NOTE: Padded positions in entity_targets should already be set to -100 (ignore_index)
        by the caller before invoking this function. This function does NOT handle padding masks.
        
        Args:
            entity_logits: (B, vocab_size_entity) or (B, L, vocab_size_entity)
            entity_targets: (B,) or (B, L) entity targets. Padded positions must be -100.
            player_civ: (B,) player civilization IDs for civ-entity masking
            
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
                batch_mask = batch_mask.unsqueeze(1).expand(B, L, V).clone()
            else:
                batch_mask = batch_mask.clone()
            
            # CRITICAL: Ensure target classes are NEVER masked out.
            # If a target entity gets masked to -inf, the loss becomes inf/NaN.
            # Only unmask for valid targets (not ignore_index=-100).
            valid_targets = entity_targets != -100
            if entity_logits.dim() == 3:
                # (B, L, V) - scatter unmask at target positions
                safe_targets = entity_targets.clone()
                safe_targets[~valid_targets] = 0  # safe index for scatter
                batch_mask.scatter_(-1, safe_targets.unsqueeze(-1), True)
            else:
                # (B, V) - unmask target entity for each sample
                safe_targets = entity_targets.clone()
                safe_targets[~valid_targets] = 0
                batch_mask[torch.arange(batch_mask.size(0), device=batch_mask.device), safe_targets] = True
        
        # Mask logits with large negative value (NOT -inf to prevent NaN in softmax/log_softmax)
        if batch_mask is not None:
            entity_logits = entity_logits.masked_fill(~batch_mask, -1e4)
            
        # Get class weights
        weights = self.entity_class_weights
        
        # Custom Loss Calculation
        if self.use_focal_loss:
            # FocalLoss returns per-element losses (reduction='none'), shape (B*L,) or (B,)
            focal_per_elem = self.criterion(entity_logits, entity_targets)

            if entity_logits.dim() == 3:
                B, L, _ = entity_logits.shape
                focal_per_pos = focal_per_elem.reshape(B, L)          # (B, L)
                valid_mask_f = (entity_targets != -100).float()        # (B, L)
                if pos_weights is not None:
                    effective_weights = valid_mask_f * pos_weights
                else:
                    effective_weights = valid_mask_f
                entity_loss = (focal_per_pos * effective_weights).sum() / effective_weights.sum().clamp(min=1.0)
            else:
                # 1-D case: no positional weighting possible, just mean over valid
                valid_mask_f = (entity_targets != -100).float()
                entity_loss = focal_per_elem.sum() / valid_mask_f.sum().clamp(min=1.0)
            
        elif self.label_smoothing > 0:
            # Compute log probabilities
            log_probs = F.log_softmax(entity_logits, dim=-1)
            
            # Create a mask for valid targets (not ignore_index)
            valid_target_mask = entity_targets != -100
            
            # Replace ignore_index with 0 for safe indexing (will be masked out)
            safe_entity_targets = entity_targets.clone()
            safe_entity_targets[~valid_target_mask] = 0
            
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
                
                # Gather targets (use safe targets to avoid negative index crash)
                if entity_logits.dim() == 2:
                    current_targets = safe_entity_targets.unsqueeze(-1)
                else:
                    # Flatten for easier handling or keep 3D
                    current_targets = safe_entity_targets.unsqueeze(-1)
                
                # Scatter add
                src = torch.full_like(current_targets, 1.0 - epsilon, dtype=true_dist.dtype)
                true_dist.scatter_add_(-1, current_targets, src)
                
                # Zero out distribution for ignored positions
                if entity_logits.dim() == 3:
                    # (B, L) -> (B, L, 1) to broadcast over vocab dim
                    true_dist = true_dist * valid_target_mask.unsqueeze(-1).float()
                else:
                    true_dist = true_dist * valid_target_mask.unsqueeze(-1).float()
            
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
                sample_weights = weights[safe_entity_targets]
                loss_per_sample = loss_per_sample * sample_weights
            
            # Average only over valid (non-padded) positions, optionally weighted by position
            valid_mask_f = valid_target_mask.float()
            if pos_weights is not None:
                effective_weights = valid_mask_f * pos_weights
            else:
                effective_weights = valid_mask_f
            weight_sum = effective_weights.sum().clamp(min=1.0)
            entity_loss = (loss_per_sample * effective_weights).sum() / weight_sum
            
        else:
            # Standard Cross Entropy (with ignore_index for padded positions)
            if entity_logits.dim() == 2 or pos_weights is None:
                if entity_logits.dim() == 2:
                    entity_loss = F.cross_entropy(entity_logits, entity_targets, weight=weights, ignore_index=-100)
                else:
                    entity_loss = F.cross_entropy(entity_logits.transpose(1, 2), entity_targets, weight=weights, ignore_index=-100)
            else:
                # (B, L, V) with positional weighting — need per-position losses
                loss_per_pos = F.cross_entropy(
                    entity_logits.transpose(1, 2), entity_targets,
                    weight=weights, ignore_index=-100, reduction='none'
                )  # (B, L)
                valid_mask_f = (entity_targets != -100).float()
                effective_weights = valid_mask_f * pos_weights
                entity_loss = (loss_per_pos * effective_weights).sum() / effective_weights.sum().clamp(min=1.0)
        
        loss_dict = {
            'loss': entity_loss.item(),
        }
        
        return entity_loss, loss_dict
    
    def train_step(self,
              batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with augmentation and contrastive learning.
        Trains on every token in the sequence (standard LM training).

        Args:
            batch: dictionary with 'entity_ids' and optionally 'mask'

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

        # Apply augmentation with some probability
        if self.use_augmentation and self.model.training and random.random() < self.augment_prob:
            entity_ids, mask = self.augmenter.augment(entity_ids, mask)

        loss_dict = {}

        # Always train on every token: input is sequence[:-1], target is sequence[1:]
        entity_input = entity_ids[:, :-1].contiguous()   # (B, L)
        entity_targets = entity_ids[:, 1:].contiguous()  # (B, L), shifted by 1
        input_mask = mask[:, :-1].contiguous() if mask is not None else None

        # Mark padded target positions as ignore_index so they don't contribute to loss
        if mask is not None:
            target_mask = mask[:, 1:].contiguous()
            entity_targets = entity_targets.clone()
            entity_targets[~target_mask] = -100

        seq_len = entity_input.size(1)

        # Late-focus pass: with late_focus_prob, zero out the first half of targets
        # so the loss trains exclusively on the latter half of the sequence (Q3-Q5).
        if torch.rand(1).item() < self.late_focus_prob:
            late_start = seq_len // 2
            entity_targets = entity_targets.clone()
            entity_targets[:, :late_start] = -100

        contrastive_emb = None
        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
            if self.use_contrastive:
                entity_logits, contrastive_emb = self.model(
                    entity_sequence=entity_input,
                    player_civ=player_civ,
                    enemy_civ=enemy_civ,
                    map_id=map_id,
                    attention_mask=input_mask,
                    predict_next=False,
                    return_embeddings=True
                )
            else:
                entity_logits = self.model(
                    entity_sequence=entity_input,
                    player_civ=player_civ,
                    enemy_civ=enemy_civ,
                    map_id=map_id,
                    attention_mask=input_mask,
                    predict_next=False
                )

            # Positional weights: linear ramp 1.0 → pos_weight_max
            pos_indices = torch.arange(seq_len, dtype=torch.float32, device=self.device)
            pos_weight_vec = 1.0 + (self.pos_weight_max - 1.0) * pos_indices / max(seq_len - 1, 1)
            pos_weights = pos_weight_vec.unsqueeze(0).expand(entity_input.size(0), -1)  # (B, L)

            # Compute main prediction loss (inside autocast for numerical stability)
            main_loss, main_loss_dict = self.compute_loss(
                entity_logits, entity_targets, player_civ, pos_weights=pos_weights
            )
            
            total_loss = main_loss
            loss_dict['main_loss'] = main_loss.item()
            
            # Add contrastive loss if enabled
            if self.use_contrastive and contrastive_emb is not None:
                contrastive_loss_val = self.contrastive_loss(contrastive_emb, player_civ)
                total_loss = total_loss + self.contrastive_weight * contrastive_loss_val
                loss_dict['contrastive_loss'] = contrastive_loss_val.item()
            
            # Add MoE auxiliary loss if model has it
            if hasattr(self.model, 'get_moe_aux_loss'):
                moe_loss = self.model.get_moe_aux_loss()
                if moe_loss.item() > 0:
                    total_loss = total_loss + moe_loss
                    loss_dict['moe_aux_loss'] = moe_loss.item()
        
        # Check for NaN/Inf loss before backward pass
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite loss detected ({total_loss.item():.4f}), skipping backward pass")
            loss_dict['loss'] = float('nan')
            return loss_dict
        
        loss_dict['loss'] = total_loss.item()
        
        # Normalize loss for gradient accumulation
        total_loss = total_loss / self.grad_accum_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Gradient accumulation
        self._accum_step += 1
        if self._accum_step >= self.grad_accum_steps:
            # Gradient clipping for stability
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
        Evaluate model on validation/test set with comprehensive metrics.
        
        Args:
            dataloader: validation/test dataloader
            
        Returns:
            metrics dictionary with accuracy, top-k accuracy, per-category metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_sequences = 0   # for loss averaging
        total_correct_entity = 0
        total_correct_top3 = 0
        total_correct_top5 = 0
        total_correct_top10 = 0
        total_samples = 0     # total valid token positions, for accuracy averaging
        
        # Track per-category performance
        category_correct = {}
        category_total = {}

        # Track accuracy for each fifth of the sequence
        num_fifths = 5
        fifth_correct = [0 for _ in range(num_fifths)]
        fifth_total = [0 for _ in range(num_fifths)]
        
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
                
                # Compute targets correctly: use the token AFTER the last valid position
                # (not entity_ids[:, -1] which is PAD for padded sequences)
                if mask is not None:
                    seq_lengths = mask.sum(dim=1).long() - 1
                    max_valid_idx = entity_ids.size(1) - 2
                    seq_lengths = torch.clamp(seq_lengths, min=0, max=max_valid_idx)
                    batch_indices = torch.arange(entity_ids.size(0), device=self.device)
                    entity_targets = entity_ids[batch_indices, seq_lengths + 1].contiguous()
                else:
                    entity_targets = entity_ids[:, -1].contiguous()
                
                entity_input = entity_ids[:, :-1].contiguous()
                
                if mask is not None:
                    input_mask = mask[:, :-1].contiguous()
                else:
                    input_mask = None
                
                # Forward pass for loss only (predict_next=True)
                entity_logits = self.model(
                    entity_sequence=entity_input,
                    player_civ=player_civ,
                    enemy_civ=enemy_civ,
                    map_id=map_id,
                    attention_mask=input_mask,
                    predict_next=True
                )
                loss, _ = self.compute_loss(entity_logits, entity_targets, player_civ)
                total_loss += loss.item() * entity_ids.size(0)
                total_sequences += entity_ids.size(0)

                # Forward pass for all accuracy metrics (predict_next=False, teacher-forced)
                # This gives predictions at every position so per-fifth and overall
                # accuracy are computed with the same protocol and are directly comparable.
                all_pos_logits = self.model(
                    entity_sequence=entity_input,
                    player_civ=player_civ,
                    enemy_civ=enemy_civ,
                    map_id=map_id,
                    attention_mask=input_mask,
                    predict_next=False
                )  # (B, L, vocab_size)

                seq_len = entity_input.size(1)
                fifth_size = seq_len // num_fifths

                # Targets at every position: entity_ids shifted by 1  (B, L)
                pos_targets = entity_ids[:, 1:seq_len + 1]

                # Valid positions: non-PAD, non-ignore targets, within the sequence mask
                pos_valid = (pos_targets != 0) & (pos_targets != -100)
                if mask is not None:
                    valid_lens = mask.sum(dim=1).long() - 1  # exclude BOS
                    for b in range(entity_input.size(0)):
                        pos_valid[b, valid_lens[b]:] = False

                all_pos_preds = torch.argmax(all_pos_logits, dim=-1)  # (B, L)
                pos_correct = (all_pos_preds == pos_targets) & pos_valid  # (B, L)

                # Overall accuracy (per token, consistent with per-fifth)
                total_correct_entity += pos_correct.sum().item()
                total_samples += pos_valid.sum().item()

                # Top-K accuracy
                _, top3_preds = all_pos_logits.topk(3, dim=-1)   # (B, L, 3)
                _, top5_preds = all_pos_logits.topk(5, dim=-1)   # (B, L, 5)
                _, top10_preds = all_pos_logits.topk(10, dim=-1) # (B, L, 10)
                tgt_exp = pos_targets.unsqueeze(-1)  # (B, L, 1)
                total_correct_top3  += (top3_preds.eq(tgt_exp).any(dim=-1)  & pos_valid).sum().item()
                total_correct_top5  += (top5_preds.eq(tgt_exp).any(dim=-1)  & pos_valid).sum().item()
                total_correct_top10 += (top10_preds.eq(tgt_exp).any(dim=-1) & pos_valid).sum().item()

                # Per-category accuracy
                for b in range(entity_input.size(0)):
                    for pos in range(seq_len):
                        if not pos_valid[b, pos]:
                            continue
                        target_id = pos_targets[b, pos].item()
                        if target_id not in category_total:
                            category_total[target_id] = 0
                            category_correct[target_id] = 0
                        category_total[target_id] += 1
                        if pos_correct[b, pos]:
                            category_correct[target_id] += 1

                # Per-fifth accuracy (vectorized)
                for f in range(num_fifths):
                    start = f * fifth_size
                    end = (f + 1) * fifth_size if f < num_fifths - 1 else seq_len
                    fifth_total[f]   += pos_valid[:, start:end].sum().item()
                    fifth_correct[f] += pos_correct[:, start:end].sum().item()
        
        avg_loss = total_loss / total_sequences
        entity_accuracy = total_correct_entity / total_samples
        entity_top3_accuracy = total_correct_top3 / total_samples
        entity_top5_accuracy = total_correct_top5 / total_samples
        entity_top10_accuracy = total_correct_top10 / total_samples
        
        # Calculate mean per-class accuracy (more robust for imbalanced data)
        per_class_acc = []
        for target_id in category_total:
            if category_total[target_id] > 0:
                per_class_acc.append(category_correct[target_id] / category_total[target_id])
        mean_per_class_acc = np.mean(per_class_acc) if per_class_acc else 0.0

        # Calculate per-fifth accuracy
        fifth_accuracies = [(fifth_correct[i] / fifth_total[i]) if fifth_total[i] > 0 else 0.0 for i in range(num_fifths)]
        
        return {
            'loss': avg_loss,
            'entity_accuracy': entity_accuracy,
            'entity_top3_accuracy': entity_top3_accuracy,
            'entity_top5_accuracy': entity_top5_accuracy,
            'entity_top10_accuracy': entity_top10_accuracy,
            'mean_per_class_accuracy': mean_per_class_acc,
            'per_fifth_accuracy': fifth_accuracies,
        }


class SequenceDataset(Dataset):
    """Dataset for entity sequence prediction with game-level splitting support."""
    def __init__(self, csv_path: str, entity_vocab: Dict[str, int], 
                 civ_vocab: Dict[str, int],
                 map_vocab: Dict[str, int], max_seq_len: int = 100,
                 filter_events: List[str] = None, filter_entities: List[str] = None,
                 only_game_start: bool = True, wins_only: bool = False):
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
            wins_only: If True, only include games where the player won (player_won == 1)
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
        
        # Filter to only include games where the player won
        if wins_only and 'player_won' in self.df.columns:
            pre_filter_len = len(self.df)
            self.df = self.df[self.df['player_won'] == 1]
            filtered_len = len(self.df)
            print(f"Filtered to wins only: {pre_filter_len - filtered_len} rows removed ({100*(pre_filter_len-filtered_len)/pre_filter_len:.1f}%), {filtered_len} rows remaining")
        elif wins_only:
            print("Warning: wins_only=True but 'player_won' column not found in data")
        
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
                       filter_entities: List[str] = None,
                       wins_only: bool = False) -> Tuple[DataLoader, DataLoader]:
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
        wins_only: If True, only include games where the player won
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    dataset = SequenceDataset(csv_path, entity_vocab, civ_vocab, map_vocab, 
                              max_seq_len, filter_events=filter_events,
                              filter_entities=filter_entities, wins_only=wins_only)
    
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
    parser = argparse.ArgumentParser(description='Train Enhanced Sequence Predictor')
    parser.add_argument('--csv_path', type=str, default='transformer_input_new.csv')
    parser.add_argument('--epochs', type=int, default=70,
                       help='Number of training epochs (more for better convergence)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (scaled for 24GB VRAM)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=16,
                       help='Number of attention heads')
    parser.add_argument('--num_decoder_layers', type=int, default=8,
                       help='Number of decoder transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='FFN dimension (4x d_model)')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout rate (lower for larger model)')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                       help='Gradient accumulation steps (effective batch = 256)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (higher for better generalization)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--warmup_epochs', type=int, default=8,
                       help='Number of warmup epochs for learning rate')
    
    # Enhanced model arguments
    parser.add_argument('--num_experts', type=int, default=8,
                       help='Number of experts in MoE layers')
    parser.add_argument('--use_moe', action=argparse.BooleanOptionalAction, default=False,
                   help='Use Mixture of Experts')
    parser.add_argument('--use_ngram', action='store_true', default=True,
                       help='Use N-gram feature extraction')
    parser.add_argument('--use_rope', action='store_true', default=True,
                       help='Use Rotary Position Embeddings')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Use contrastive learning auxiliary loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.05,
                       help='Weight for contrastive loss (lower to focus on main task)')
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                       help='Use sequence augmentation during training')
    parser.add_argument('--augment_prob', type=float, default=0.2,
                       help='Probability of applying augmentation (lower for stability)')
    parser.add_argument('--pos_weight_max', type=float, default=3.0,
                       help='Max positional loss weight (position 0 = 1.0, position L-1 = this value)')
    parser.add_argument('--late_focus_prob', type=float, default=0.4,
                       help='Fraction of batches that train exclusively on the latter half of the sequence')
    
    # WandB arguments
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity')
    parser.add_argument('--wandb_project', type=str, default='Build_Order_Prediction',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wins_only', action='store_true',
                       help='Only train on games where the player won')
    
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
        filter_entities=filter_entities,
        wins_only=args.wins_only
    )
    
    # Create model with conditional decoder architecture
    # NOTE: This is NOT an encoder-decoder. The condition projection is just an MLP.
    # Self-attention on a single token (the condition vector) is useless, so we removed it.
    print("Initializing conditional decoder model...")
    model = SequencePredictor(
        vocab_size_entity=len(entity_vocab),
        civ_vocab_size=len(civ_vocab),
        map_vocab_size=len(map_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_experts=args.num_experts,
        use_moe=args.use_moe,
        use_ngram=args.use_ngram,
        use_rope=args.use_rope
    ).to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
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
    
    # Create optimizer with weight decay (separate learning rates for different components)
    # Use lower LR for embeddings, higher for attention
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'embed' in n], 'lr': args.lr * 0.3},
        {'params': [p for n, p in model.named_parameters() if 'embed' not in n], 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0.05,  # Lower weight decay for larger model (0.05 is standard for ViT/BERT)
        betas=(0.9, 0.98),
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
        # Cosine decay after warmup with minimum LR
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Min LR = 1% of max
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create trainer with enhanced features
    trainer = SequencePredictorTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_amp=True,
        grad_accum_steps=args.grad_accum_steps,
        civ_entity_mask=civ_entity_mask,
        entity_class_weights=entity_class_weights,
        label_smoothing=args.label_smoothing,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight,
        use_augmentation=args.use_augmentation,
        augment_prob=args.augment_prob,
        pos_weight_max=args.pos_weight_max,
        late_focus_prob=args.late_focus_prob
    )
    
    # Initialize Weights & Biases
    if not args.no_wandb:
        wb_cfg = {
            'learning_rate': args.lr,
            'architecture': 'ConditionalDecoder',  # NOT encoder-decoder! Just MLP projection + decoder
            'dataset': os.path.basename(args.csv_path),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * args.grad_accum_steps,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_decoder_layers': args.num_decoder_layers,
            'dim_feedforward': args.dim_feedforward,
            'max_seq_len': args.max_seq_len,
            'dropout': args.dropout,
            'grad_accum_steps': args.grad_accum_steps,
            'val_split': args.val_split,
            'entity_vocab_size': len(entity_vocab),
            'civ_vocab_size': len(civ_vocab),
            'map_vocab_size': len(map_vocab),
            'warmup_epochs': args.warmup_epochs,
            'label_smoothing': args.label_smoothing,
            'weight_decay': 0.1,
            'total_params': total_params,
            'trainable_params': trainable_params,
            # Enhanced architecture flags
            'use_moe': args.use_moe,
            'num_experts': args.num_experts,
            'use_ngram': args.use_ngram,
            'use_rope': args.use_rope,
            'use_contrastive': args.use_contrastive,
            'contrastive_weight': args.contrastive_weight,
            'use_augmentation': args.use_augmentation,
            'augment_prob': args.augment_prob,
            'pos_weight_max': args.pos_weight_max,
            'late_focus_prob': args.late_focus_prob,
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
    patience = 10  # Early stopping patience (longer for 100 epochs)
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        train_losses = []
        epoch_grad_norm = 0.0
        grad_norm_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss_dict = trainer.train_step(batch)
            
            # Only track finite losses (skip NaN from skipped batches)
            if math.isfinite(loss_dict['loss']):
                train_losses.append(loss_dict['loss'])
            
            # Step scheduler and compute grad norm after each optimizer step
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Compute gradient norm BEFORE optimizer.zero_grad() clears them
                # (This is done inside train_step now, but we capture it here
                # for logging since train_step has already stepped the optimizer)
                # We compute it from the model params that still have grad from
                # the last backward pass before optimizer stepped
                current_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        current_grad_norm += p.grad.data.norm(2).item() ** 2
                current_grad_norm = current_grad_norm ** 0.5
                epoch_grad_norm += current_grad_norm
                grad_norm_count += 1
                
                scheduler.step()
                global_step += 1
            
            
            # Log batch progress
            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss_dict['loss']:.4f}, LR: {current_lr:.6f}")
        
        # Calculate training metrics (only from finite losses)
        if train_losses:
            avg_train_loss = np.mean(train_losses)
        else:
            avg_train_loss = float('nan')
            print("  WARNING: All training losses were NaN this epoch!")
        
        print(f"\nTraining Results:")
        print(f"  Average Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("\nRunning validation...")
        val_metrics = trainer.evaluate(val_loader)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Entity Accuracy: {val_metrics['entity_accuracy']:.4f} ({val_metrics['entity_accuracy']*100:.1f}%)")
        print(f"  Top-3 Accuracy: {val_metrics['entity_top3_accuracy']:.4f} ({val_metrics['entity_top3_accuracy']*100:.1f}%)")
        print(f"  Top-5 Accuracy: {val_metrics['entity_top5_accuracy']:.4f} ({val_metrics['entity_top5_accuracy']*100:.1f}%)")
        print(f"  Top-10 Accuracy: {val_metrics['entity_top10_accuracy']:.4f} ({val_metrics['entity_top10_accuracy']*100:.1f}%)")
        print(f"  Mean Per-Class Accuracy: {val_metrics['mean_per_class_accuracy']:.4f}")
        fifth_accs = val_metrics.get('per_fifth_accuracy', [])
        if fifth_accs:
            fifth_str = '  |  '.join([f"Q{i+1}: {a*100:.1f}%" for i, a in enumerate(fifth_accs)])
            print(f"  Per-Fifth Accuracy: {fifth_str}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Use composite metric for model selection (weighted accuracy + loss)
        composite_score = val_metrics['entity_accuracy'] * 0.5 + val_metrics['entity_top5_accuracy'] * 0.3 - val_metrics['loss'] * 0.2
        
        # Save best model (based on accuracy rather than just loss)
        if val_metrics['entity_accuracy'] > getattr(main, 'best_accuracy', 0) or \
           (val_metrics['entity_accuracy'] == getattr(main, 'best_accuracy', 0) and val_metrics['loss'] < best_val_loss):
            main.best_accuracy = val_metrics['entity_accuracy']
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
                'val_top5_accuracy': val_metrics['entity_top5_accuracy'],
                'val_top10_accuracy': val_metrics['entity_top10_accuracy'],
                'entity_vocab': entity_vocab,
                'civ_vocab': civ_vocab,
                'map_vocab': map_vocab,
                'civ_entity_mapping': {k: list(v) for k, v in civ_entity_mapping.items()},
                'args': vars(args)
            }
            
            torch.save(checkpoint, 'best_model.pth')
            print(f"  ✓ Saved best model (accuracy: {val_metrics['entity_accuracy']:.4f}, loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        # Log to Weights & Biases
        if not args.no_wandb:
            # Use average gradient norm computed DURING training (not after eval)
            avg_grad_norm = epoch_grad_norm / max(grad_norm_count, 1)
            wandb_log_dict = {
                'epoch': epoch + 1,
                # Training metrics
                'train/loss': avg_train_loss,
                'train/learning_rate': current_lr,
                'train/grad_norm': avg_grad_norm,
                # Validation metrics
                'val/loss': val_metrics['loss'],
                'val/entity_accuracy': val_metrics['entity_accuracy'],
                'val/entity_top3_accuracy': val_metrics['entity_top3_accuracy'],
                'val/entity_top5_accuracy': val_metrics['entity_top5_accuracy'],
                'val/entity_top10_accuracy': val_metrics['entity_top10_accuracy'],
                'val/mean_per_class_accuracy': val_metrics['mean_per_class_accuracy'],
                # Per-fifth-of-sequence accuracy
                **{f'val/per_fifth_accuracy_{i+1}': acc
                   for i, acc in enumerate(val_metrics.get('per_fifth_accuracy', []))},
                # Best metrics tracking
                'best/val_loss': best_val_loss,
                'best/val_accuracy': getattr(main, 'best_accuracy', 0),
                'analysis/train_val_loss_gap': avg_train_loss - val_metrics['loss'],
                'training/patience_counter': patience_counter,
            }
            wandb.log(wandb_log_dict, step=epoch + 1)
            # Log example predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_example_predictions(model, val_loader, entity_vocab, device, wandb)
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = trainer.evaluate(val_loader)
    print(f"\nFinal Validation Results (best model):")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Entity Accuracy: {final_metrics['entity_accuracy']:.4f} ({final_metrics['entity_accuracy']*100:.1f}%)")
    print(f"  Top-3 Accuracy: {final_metrics['entity_top3_accuracy']:.4f} ({final_metrics['entity_top3_accuracy']*100:.1f}%)")
    print(f"  Top-5 Accuracy: {final_metrics['entity_top5_accuracy']:.4f} ({final_metrics['entity_top5_accuracy']*100:.1f}%)")
    print(f"  Top-10 Accuracy: {final_metrics['entity_top10_accuracy']:.4f} ({final_metrics['entity_top10_accuracy']*100:.1f}%)")
    print(f"  Mean Per-Class Accuracy: {final_metrics['mean_per_class_accuracy']:.4f}")
    
    # Save final model with additional metadata
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'entity_vocab': entity_vocab,
        'civ_vocab': civ_vocab,
        'map_vocab': map_vocab,
        'civ_entity_mapping': {k: list(v) for k, v in civ_entity_mapping.items()},
        'args': vars(args),
        'val_metrics': final_metrics,
        'best_val_loss': best_val_loss,
        'architecture': 'ConditionalDecoder'  # NOT encoder-decoder! Just MLP condition projection + decoder
    }
    torch.save(final_checkpoint, 'final_model.pth')
    
    # Log final results to wandb
    if not args.no_wandb:
        wandb.log({
            'final/val_loss': final_metrics['loss'],
            'final/entity_accuracy': final_metrics['entity_accuracy'],
            'final/entity_top3_accuracy': final_metrics['entity_top3_accuracy'],
            'final/entity_top5_accuracy': final_metrics['entity_top5_accuracy'],
            'final/entity_top10_accuracy': final_metrics['entity_top10_accuracy'],
            'final/mean_per_class_accuracy': final_metrics['mean_per_class_accuracy'],
        })
        
        # Save model to wandb
        wandb.save('best_model.pth')
        wandb.save('final_model.pth')
        
        # Finish wandb run
        wandb.finish()
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Best Accuracy: {final_metrics['entity_accuracy']*100:.1f}%")
    print(f"  Best Top-5 Accuracy: {final_metrics['entity_top5_accuracy']*100:.1f}%")
    print(f"  Model saved as 'final_model.pth'")
    print(f"{'='*60}")


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
    """Compute class weights for entity loss using log-dampened inverse frequency.
    
    This function uses pure frequency-based weighting without manual category overrides,
    letting the model learn entity importance from the data distribution.
    
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
    
    # === Compute Weights Using Log-Dampened Inverse Frequency ===
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
    
    # === Cap Extreme Weights ===
    # Very rare entities shouldn't dominate the loss
    valid_mask = weights > 0
    if valid_mask.sum() > 0:
        # Cap at 95th percentile to prevent outlier domination
        cap_value = torch.quantile(weights[valid_mask], 0.95).item()
        weights = torch.clamp(weights, max=cap_value)
    
    # === Normalize to Mean = 1.0 ===
    valid_mask = weights > 0
    if valid_mask.sum() > 0:
        mean_weight = weights[valid_mask].mean()
        weights[valid_mask] = weights[valid_mask] / mean_weight
    
    # === Print Summary ===
    print(f"\n{'='*60}")
    print(f"Entity Class Weights Summary (Log-Dampened Inverse Frequency)")
    print(f"{'='*60}")
    print(f"  Total entities: {len(entity_counts)}")
    print(f"  Min weight: {weights[valid_mask].min():.4f}")
    print(f"  Max weight: {weights[valid_mask].max():.4f}")
    print(f"  Mean weight: {weights[valid_mask].mean():.4f}")
    print(f"  Median weight: {torch.median(weights[valid_mask]):.4f}")
    
    # Show weight distribution
    inv_vocab = {v: k for k, v in entity_vocab.items()}
    weight_list = [(inv_vocab[i], weights[i].item()) for i in range(vocab_size) if weights[i] > 0]
    weight_list.sort(key=lambda x: x[1])
    
    print(f"\n  Lowest weights (high-frequency entities):")
    for name, w in weight_list[:8]:
        count = entity_counts.get(name, 0)
        print(f"    {name}: {w:.4f} (count: {count:,})")
    
    print(f"\n  Highest weights (low-frequency entities):")
    for name, w in weight_list[-8:]:
        count = entity_counts.get(name, 0)
        print(f"    {name}: {w:.4f} (count: {count:,})")
    
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
            entity_input, player_civ, enemy_civ, map_id, 
            attention_mask=input_mask, predict_next=True
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