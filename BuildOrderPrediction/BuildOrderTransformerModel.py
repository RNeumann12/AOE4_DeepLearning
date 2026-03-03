import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional

from WinRatePrediction.WinRateTransformerModel import AoETransformer, TimePositionalEncoding


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This helps the model focus on hard examples (rare entities like military units)
    instead of easy examples (common entities like Villagers).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean', ignore_index: int = 0, 
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, C) or (B, L, C) logits
            target: (B,) or (B, L) indices
        """
        # Flatten if 3D (sequence model output)
        if input.ndim > 2:
            c = input.shape[-1]
            input = input.reshape(-1, c)
            target = target.reshape(-1)

        # Cast to float32 for numerical stability
        input = input.float()
        
        # Create mask for valid (non-ignored) indices
        valid_mask = target != self.ignore_index
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        # Filter to valid indices only
        input_valid = input[valid_mask]
        target_valid = target[valid_mask]
        
        # Apply label smoothing if specified
        num_classes = input_valid.size(-1)
        if self.label_smoothing > 0:
            # Create smoothed target distribution
            smooth_target = torch.zeros_like(input_valid)
            smooth_target.fill_(self.label_smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target_valid.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Compute log probabilities
            log_probs = F.log_softmax(input_valid, dim=-1)
            probs = torch.exp(log_probs)
            
            # For focal loss modulation, use the probability of the true class
            pt = probs.gather(1, target_valid.unsqueeze(1)).squeeze(1)
            
            # Compute smoothed cross entropy
            ce_loss = -(smooth_target * log_probs).sum(dim=-1)
        else:
            # Standard cross entropy
            log_probs = F.log_softmax(input_valid, dim=-1)
            probs = torch.exp(log_probs)
            pt = probs.gather(1, target_valid.unsqueeze(1)).squeeze(1)
            ce_loss = F.nll_loss(log_probs, target_valid, reduction='none')
        
        # Focal modulation: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            alpha = self.alpha.to(input.device)
            alpha_t = alpha[target_valid]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BuildOrderGenerator(nn.Module):
    """
    Transformer-based build order generator.
    Uses encoder-decoder architecture with causal masking for generation.
    """
    def __init__(self,
                 vocab_size_entity: int,
                 vocab_size_event: int,
                 civ_vocab_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_len: int = 1024,
                 max_time_seconds: int = 5400):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Shared Embeddings
        self.entity_embed = nn.Embedding(vocab_size_entity, d_model)
        self.event_embed = nn.Embedding(vocab_size_event, d_model)
        self.civ_embed = nn.Embedding(civ_vocab_size, d_model)
        
        # Positional Encodings
        self.seq_pos_embed = nn.Embedding(max_len, d_model)
        self.time_encoding = TimePositionalEncoding(d_model, max_len=max_time_seconds)
        
        # 2. Encoder (for conditioning on civ matchup)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 3. Decoder (for generating build order)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 4. Output Heads
        self.entity_head = nn.Linear(d_model, vocab_size_entity)
        self.event_head = nn.Linear(d_model, vocab_size_event)
        self.time_head = nn.Linear(d_model, 1)  # Predict time delta
        
        # 5. Conditioning Projection
        self.condition_proj = nn.Linear(d_model * 3, d_model)  # Player civ + enemy civ + win prob

        # Projection to combine entity/event/pos embeddings before decoding
        self.combine_proj = nn.Linear(d_model * 3, d_model)

        # Normalization layer applied to combined decoder inputs
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # 6. Win Probability Head (optional, can use pretrained model)
        self.win_prob_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 7. Condition Embedding (Loss=0, Win=1, Unknown=2)
        # We replace the float projection with a learnable embedding to avoid 0.0 (Loss) vs None (Unknown) ambiguity
        self.win_condition_embed = nn.Embedding(3, d_model)
        
        # Special tokens (include sequence dim so they can be expanded to (B,1,D))
        self.start_token_entity = nn.Parameter(torch.randn(1, 1, d_model))
        self.start_token_event = nn.Parameter(torch.randn(1, 1, d_model))
        with torch.no_grad():
            # Initialize start tokens from the mean embedding (shape: (1,1,d_model))
            self.start_token_entity.copy_(self.entity_embed.weight.mean(dim=0, keepdim=True).unsqueeze(1))
            self.start_token_event.copy_(self.event_embed.weight.mean(dim=0, keepdim=True).unsqueeze(1))

        # Mask sentinel value (fp16-safe) used when masking logits with AMP enabled
        # Use a large negative number that fits in float16 range (max ~ -65504). We pick -1e4.
        self.mask_value = float(-1e4)
        
    def encode_condition(self, player_civ: torch.Tensor, enemy_civ: torch.Tensor, 
                         target_win_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the condition (civ matchup + optional target win probability)
        """
        batch_size = player_civ.size(0)
        device = player_civ.device
        
        # Embed civilizations
        p_civ = self.civ_embed(player_civ)  # (B, d_model)
        e_civ = self.civ_embed(enemy_civ)    # (B, d_model)
        
        # Embed win condition
        if target_win_prob is not None:
            # Discretize: <0.5 -> Loss (0), >=0.5 -> Win (1)
            # Input is expected to be probabilities 0.0 to 1.0
            cond_indices = (target_win_prob >= 0.5).long()
        else:
            # Unknown -> 2
            cond_indices = torch.full((batch_size,), 2, dtype=torch.long, device=device)
            
        win_embed = self.win_condition_embed(cond_indices) # (B, d_model)
        
        # Concatenate: [p_civ, e_civ, win_embed]
        condition = torch.cat([p_civ, e_civ, win_embed], dim=1)
            
        # Project to d_model
        condition = self.condition_proj(condition)  # (B, d_model)
        
        # Add sequence dimension and encode
        condition = condition.unsqueeze(1)  # (B, 1, d_model)
        condition = self.encoder(condition)
        
        return condition  # (B, 1, d_model)

    def set_civ_entity_mask(self, mask: torch.Tensor):
        """Set a civ->entity mask to restrict which entities are valid per civilization.

        Args:
            mask: bool Tensor of shape (num_civs_plus_one, vocab_size_entity) where index corresponds
                  to civilization id (including padding 0). True indicates entity is allowed.
        """
        if mask.dim() != 2:
            raise ValueError("civ entity mask must be 2-dimensional (num_civs, vocab_size_entity)")
        # Register as buffer so it moves to device with the model
        self.register_buffer('civ_entity_mask', mask.bool())
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filter logits to keep only top-k values, setting others to -inf."""
        if k <= 0:
            return logits
        # logits shape: (B, 1, V) or (B, V)
        values, _ = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
        min_value = values[..., -1:]
        return torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)

    def forward(self, 
                player_civ: torch.Tensor,
                enemy_civ: torch.Tensor,
                target_sequence_length: int,
                target_win_prob: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5,
                ground_truth: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                allowed_entities_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate build order or train with teacher forcing.

        Args:
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            target_sequence_length: length of build order to generate
            target_win_prob: (B,) target win probability (0-1), optional
            teacher_forcing_ratio: probability of using ground truth during training
            ground_truth: tuple of (entity_ids, event_ids, times) for teacher forcing
            allowed_entities_mask: optional (B, V_entity) boolean mask specifying which entity ids are valid
                                   for each example in the batch. If not provided and a civ mask was set via
                                   `set_civ_entity_mask`, the mask will be derived from the player's civ id.

        Returns:
            predicted_entities: (B, L, vocab_size_entity)
            predicted_events: (B, L, vocab_size_event)
            predicted_times: (B, L)
            predicted_win_logits: (B,)  # raw logits (use BCEWithLogitsLoss)
        """        """
        Generate build order or train with teacher forcing.
        
        Args:
            player_civ: (B,) player civilization IDs
            enemy_civ: (B,) enemy civilization IDs
            target_sequence_length: length of build order to generate
            target_win_prob: (B,) target win probability (0-1), optional
            teacher_forcing_ratio: probability of using ground truth during training
            ground_truth: tuple of (entity_ids, event_ids, times) for teacher forcing
            
        Returns:
            predicted_entities: (B, L, vocab_size_entity)
            predicted_events: (B, L, vocab_size_event)
            predicted_times: (B, L)
            predicted_win_logits: (B,)  # raw logits (use BCEWithLogitsLoss)
        """
        batch_size = player_civ.size(0)
        device = player_civ.device
        
        # Encode condition
        memory = self.encode_condition(player_civ, enemy_civ, target_win_prob)  # (B, 1, d_model)
        
        # Initialize with start tokens
        current_entities = self.start_token_entity.expand(batch_size, 1, -1)  # (B, 1, d_model)
        current_events = self.start_token_event.expand(batch_size, 1, -1)  # (B, 1, d_model)
        
        # Store predictions
        all_entity_logits = []
        all_event_logits = []
        all_time_preds = []

        # ============================================
        # PARALLEL MODE (teacher_forcing_ratio == 1.0 with ground truth)
        # Used for training AND evaluation with teacher forcing
        # ============================================
        if teacher_forcing_ratio == 1.0 and ground_truth is not None:
            entity_targets, event_targets, time_targets = ground_truth
            seq_len = entity_targets.size(1)
            batch_size = entity_targets.size(0)

            # Build shifted decoder inputs: [START] + ground_truth[:, :-1]
            if seq_len > 1:
                entity_input_emb = torch.cat([
                    self.start_token_entity.expand(batch_size, 1, -1),
                    self.entity_embed(entity_targets[:, :-1])
                ], dim=1)
                event_input_emb = torch.cat([
                    self.start_token_event.expand(batch_size, 1, -1),
                    self.event_embed(event_targets[:, :-1])
                ], dim=1)
            else:
                entity_input_emb = self.start_token_entity.expand(batch_size, 1, -1)
                event_input_emb = self.start_token_event.expand(batch_size, 1, -1)

            positions = torch.arange(0, entity_input_emb.size(1), device=device).unsqueeze(0).expand(batch_size, entity_input_emb.size(1))
            pos_emb = self.seq_pos_embed(positions)
            # Combine via normalization of summed embeddings (stabilizes training when teacher forcing)
            decoder_input = self.decoder_norm(entity_input_emb + event_input_emb + pos_emb)

            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            decoder_output = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)

            # Compute logits for all positions in parallel
            entity_logits = self.entity_head(decoder_output)  # (B, L, V)
            event_logits = self.event_head(decoder_output)    # (B, L, V)
            time_preds = torch.relu(self.time_head(decoder_output)).squeeze(-1)  # (B, L)

            # Store unmasked logits for loss computation (label smoothing requires all classes accessible)
            entity_logits_unmasked = entity_logits

            # Derive allowed mask if not provided
            if allowed_entities_mask is None and hasattr(self, 'civ_entity_mask') and self.civ_entity_mask is not None:
                allowed = self.civ_entity_mask[player_civ]
            else:
                allowed = allowed_entities_mask

            # Apply civ-based mask if provided (only affects sampling/argmax, not loss)
            if allowed is not None:
                allowed_b1v = allowed.unsqueeze(1)  # (B, 1, V)
                entity_logits = entity_logits.masked_fill(~allowed_b1v, self.mask_value)

            # Predict win prob from final decoder output
            final_output = decoder_output[:, -1, :]
            win_logits = self.win_prob_head(final_output).squeeze(-1)

            # Return unmasked logits for training loss, masked logits would cause label smoothing issues
            # (masked classes get -1e4 logits -> softmax ~= 0 -> log(0) in smoothed loss = inf)
            return entity_logits_unmasked, event_logits, time_preds, win_logits

        # Generate sequence autoregressively
        for step in range(target_sequence_length):
            # Current sequence length (including start token + generated steps)
            seq_len = step + 1
            
            # Create positional embeddings for current sequence
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            pos_emb = self.seq_pos_embed(positions)  # (B, seq_len, d_model)
            
            # Combine current embeddings with positional encoding via LayerNorm on the sum
            decoder_input = self.decoder_norm(current_entities + current_events + pos_emb)
            
            # Create causal mask for decoder
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
            
            # Decode
            decoder_output = self.decoder(
                decoder_input, 
                memory,
                tgt_mask=tgt_mask,
            )  # (B, seq_len, d_model)
            
            # Get predictions for the last position only
            last_output = decoder_output[:, -1:, :]  # (B, 1, d_model)
            
            # Predict next tokens
            entity_logits = self.entity_head(last_output)  # (B, 1, vocab_size_entity)
            event_logits = self.event_head(last_output)    # (B, 1, vocab_size_event)
            time_delta = torch.nn.functional.softplus(self.time_head(last_output)).squeeze(-1)

            # Store UNMASKED logits for loss computation (label smoothing requires all classes accessible)
            # Masking with -1e4 values causes label smoothing to produce huge losses
            all_entity_logits.append(entity_logits)
            all_event_logits.append(event_logits)
            all_time_preds.append(time_delta)

            # Derive allowed mask if not provided
            if allowed_entities_mask is None and hasattr(self, 'civ_entity_mask') and self.civ_entity_mask is not None:
                # civ_entity_mask indexed by civ id -> (num_civs, vocab)
                allowed = self.civ_entity_mask[player_civ]
            else:
                allowed = allowed_entities_mask

            # Apply mask to entity logits for sampling only (not stored for loss)
            entity_logits_masked = entity_logits.clone()
            # Mask special tokens: PAD (index 0) and UNK (index 1) should never be selected
            entity_logits_masked[:, :, 0] = self.mask_value  # <PAD>
            entity_logits_masked[:, :, 1] = self.mask_value  # <UNK>
            
            # Similarly mask special tokens for events
            event_logits_masked = event_logits.clone()
            event_logits_masked[:, :, 0] = self.mask_value  # <PAD>
            event_logits_masked[:, :, 1] = self.mask_value  # <UNK>
            
            if allowed is not None:
                # allowed: (B, V) -> expand to (B, 1, V) to match logits
                allowed_b1v = allowed.unsqueeze(1)
                entity_logits_masked = entity_logits_masked.masked_fill(~allowed_b1v, self.mask_value)
            
            # Decide whether to use ground truth or predictions
            # Use teacher forcing when: ground truth is available AND random sample < teacher_forcing_ratio
            # Note: This works in both training and eval mode to allow teacher forcing evaluation
            use_teacher_forcing = ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                # Teacher forcing: use ground truth
                next_entity = ground_truth[0][:, step:step+1]  # (B, 1)
                next_event = ground_truth[1][:, step:step+1]   # (B, 1)
                next_time = ground_truth[2][:, step:step+1].unsqueeze(-1) if step > 0 else time_delta.unsqueeze(-1)
            else:
                # Temperature-scaled sampling with optional top-k filtering
                # This prevents mode collapse (always predicting Villager)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    entity_logits_scaled = entity_logits_masked / max(temperature, 1e-8)
                    event_logits_scaled = event_logits_masked / max(temperature, 1e-8)
                else:
                    entity_logits_scaled = entity_logits_masked
                    event_logits_scaled = event_logits_masked
                
                # Apply top-k filtering if specified
                if top_k > 0:
                    entity_logits_scaled = self._top_k_filter(entity_logits_scaled, top_k)
                    event_logits_scaled = self._top_k_filter(event_logits_scaled, top_k)
                
                # Sample from distribution instead of greedy argmax
                entity_probs = F.softmax(entity_logits_scaled.squeeze(1), dim=-1)  # (B, V)
                event_probs = F.softmax(event_logits_scaled.squeeze(1), dim=-1)    # (B, V)
                
                # Handle potential NaN/inf from temperature scaling
                entity_probs = torch.nan_to_num(entity_probs, nan=0.0, posinf=0.0, neginf=0.0)
                event_probs = torch.nan_to_num(event_probs, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure valid probability distributions
                entity_probs = entity_probs / (entity_probs.sum(dim=-1, keepdim=True) + 1e-8)
                event_probs = event_probs / (event_probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                next_entity = torch.multinomial(entity_probs, 1)  # (B, 1)
                next_event = torch.multinomial(event_probs, 1)    # (B, 1)
                next_time = time_delta.unsqueeze(-1)  # (B, 1)
            
            # Embed the chosen tokens for next step
            next_entity_embed = self.entity_embed(next_entity)  # (B, 1, d_model)
            next_event_embed = self.event_embed(next_event)     # (B, 1, d_model)
            
            # Append to current sequence
            current_entities = torch.cat([current_entities, next_entity_embed], dim=1)
            current_events = torch.cat([current_events, next_event_embed], dim=1)
        
        # Stack predictions
        entity_logits = torch.cat(all_entity_logits, dim=1)  # (B, L, vocab_size_entity)
        event_logits = torch.cat(all_event_logits, dim=1)    # (B, L, vocab_size_event)
        time_preds = torch.cat(all_time_preds, dim=1)        # (B, L)
        
        # Predict win probability from final decoder output (return logits for training loss)
        final_output = decoder_output[:, -1, :]  # (B, d_model)
        win_logits = self.win_prob_head(final_output).squeeze(-1)  # (B,)
        
        return entity_logits, event_logits, time_preds, win_logits
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def beam_search_generate(self,
                           player_civ: torch.Tensor,
                           enemy_civ: torch.Tensor,
                           beam_width: int = 5,
                           max_length: int = 50,
                           temperature: float = 1.0,
                           target_win_prob: Optional[torch.Tensor] = None) -> Tuple[List[Tuple], torch.Tensor]:
        """
        Generate optimal build order using beam search.
        
        Returns:
            best_sequences: list of (entities, events, times) tuples
            win_probs: corresponding win probabilities
        """
        batch_size = player_civ.size(0)
        device = player_civ.device
        
        # Encode condition once
        memory = self.encode_condition(player_civ, enemy_civ, target_win_prob)  # (B, 1, d_model)
        
        # Initialize beams for each batch item
        all_best_sequences = []
        all_win_probs = []

        for b in range(batch_size):
            # Single batch item memory (no need to expand)
            mem0 = memory[b:b+1]  # (1, 1, d_model)

            # Start with a single empty beam and expand each step
            beams = [{
                'entities': torch.empty(0, dtype=torch.long, device=device),
                'events': torch.empty(0, dtype=torch.long, device=device),
                'times': torch.empty(0, dtype=torch.float, device=device),
                'entity_embs': self.start_token_entity.expand(1, -1, -1).clone(),  # (1,1,d_model)
                'event_embs': self.start_token_event.expand(1, -1, -1).clone(),    # (1,1,d_model)
                'score': 0.0,
                'last_time': 0.0
            }]

            for step in range(max_length):
                candidates = []
                for beam in beams:
                    seq_len = beam['entities'].size(0) + 1  # +1 for start token
                    positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
                    pos_emb = self.seq_pos_embed(positions)  # (1, seq_len, d_model)

                    decoder_input = self.decoder_norm(beam['entity_embs'] + beam['event_embs'] + pos_emb)
                    tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

                    with torch.no_grad():
                        decoder_output = self.decoder(decoder_input, mem0, tgt_mask=tgt_mask)
                        last_output = decoder_output[:, -1:, :]  # (1,1,d_model)

                        entity_logits = (self.entity_head(last_output) / temperature).squeeze(0).squeeze(0)  # (vocab,)
                        event_logits = (self.event_head(last_output) / temperature).squeeze(0).squeeze(0)    # (vocab,)
                        time_delta = torch.relu(self.time_head(last_output)).squeeze()  # scalar

                        # Apply civ-based mask if available
                        if hasattr(self, 'civ_entity_mask') and self.civ_entity_mask is not None:
                            # player civ id for this batch item
                            civ_id = player_civ[b]
                            allowed = self.civ_entity_mask[civ_id].to(device)
                            entity_logits = entity_logits.masked_fill(~allowed, self.mask_value)

                        # Top-k for each head
                        k_e = min(beam_width, entity_logits.size(-1))
                        k_ev = min(beam_width, event_logits.size(-1))
                        ent_vals, ent_idx = torch.topk(torch.softmax(entity_logits, dim=-1), k=k_e)
                        ev_vals, ev_idx = torch.topk(torch.softmax(event_logits, dim=-1), k=k_ev)

                        for i in range(k_e):
                            for j in range(k_ev):
                                new_entity_id = ent_idx[i].unsqueeze(0)
                                new_event_id = ev_idx[j].unsqueeze(0)

                                new_entities = torch.cat([beam['entities'], new_entity_id])
                                new_events = torch.cat([beam['events'], new_event_id])
                                new_time = beam['last_time'] + float(time_delta.item())
                                new_times = torch.cat([beam['times'], torch.tensor([new_time], device=device)])

                                new_entity_emb = self.entity_embed(new_entity_id).unsqueeze(0)  # (1,1,d_model)
                                new_event_emb = self.event_embed(new_event_id).unsqueeze(0)
                                new_entity_embs = torch.cat([beam['entity_embs'], new_entity_emb], dim=1)
                                new_event_embs = torch.cat([beam['event_embs'], new_event_emb], dim=1)

                                score = beam['score'] + float(torch.log(ent_vals[i] + 1e-10).item()) + float(torch.log(ev_vals[j] + 1e-10).item())

                                candidates.append({
                                    'entities': new_entities,
                                    'events': new_events,
                                    'times': new_times,
                                    'entity_embs': new_entity_embs,
                                    'event_embs': new_event_embs,
                                    'score': score,
                                    'last_time': new_time
                                })

                # If no candidates (shouldn't happen), stop
                if not candidates:
                    break

                # Keep top beams
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_width]

            # Evaluate final beams using win predictor head
            final_beams = []
            for beam in beams:
                seq_len = beam['entities'].size(0) + 1
                positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
                pos_emb = self.seq_pos_embed(positions)
                decoder_input = self.decoder_norm(beam['entity_embs'] + beam['event_embs'] + pos_emb)
                tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)

                with torch.no_grad():
                    decoder_output = self.decoder(decoder_input, mem0, tgt_mask=tgt_mask)
                    final_output = decoder_output[:, -1, :]
                    win_prob = float(torch.sigmoid(self.win_prob_head(final_output)).squeeze().item())

                final_beams.append((beam, win_prob))

            final_beams.sort(key=lambda x: x[1], reverse=True)
            best_beam, best_win_prob = final_beams[0]

            all_best_sequences.append((
                best_beam['entities'].cpu().numpy(),
                best_beam['events'].cpu().numpy(),
                best_beam['times'].cpu().numpy()
            ))
            all_win_probs.append(best_win_prob)

        return all_best_sequences, torch.tensor(all_win_probs, device=device)


class BuildOrderTrainer:
    """
    Trainer for the build order generator.
    Can optionally use a pretrained win prediction model as a reward signal.

    Supports AMP (automatic mixed precision) and gradient accumulation to reduce memory usage.
    Uses Focal Loss to address class imbalance (prevents model from always predicting Villager).
    """
    def __init__(self, model, optimizer, device, use_reinforce=False, reward_model=None, 
                 use_amp: bool = False, grad_accum_steps: int = 1, label_smoothing: float = 0.1,
                 use_focal_loss: bool = True, focal_gamma: float = 2.0,
                 entity_class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_reinforce = use_reinforce
        self.reward_model = reward_model
        # Determine whether ROCm/HIP is present
        hip_present = getattr(torch.version, 'hip', None) is not None
        self.hip_present = hip_present
        # Choose AMP dtype: use bfloat16 on ROCm because float16 GEMMs can be unstable on some AMD GPUs
        if use_amp and hip_present:
            print("Note: ROCm/HIP detected; AMP will use bfloat16 if available.")
        # Enable AMP if requested and CUDA is available (or ROCm present)
        self.use_amp = use_amp and (torch.cuda.is_available() or hip_present)
        self.amp_dtype = torch.bfloat16 if hip_present else torch.float16
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._accum_step = 0
        # Device type for autocast / GradScaler: prefer 'cuda' when a GPU or ROCm is present
        self.amp_device = 'cuda' if (torch.cuda.is_available() or hip_present) else 'cpu'
        # Initialize GradScaler compatibly depending on PyTorch version
        if self.use_amp:
            try:
                # Newer API accepts device_type
                self.scaler = torch.amp.GradScaler(device_type=self.amp_device)
            except TypeError:
                # Fallbacks for older PyTorch versions
                if self.amp_device == 'cuda':
                    self.scaler = torch.cuda.amp.GradScaler()
                else:
                    # CPU GradScaler may not be supported; set to None and skip scaling
                    self.scaler = None
        else:
            self.scaler = None
        
        # Label smoothing to prevent overconfidence on common classes like Villager
        self.label_smoothing = label_smoothing
        
        # Focal Loss configuration to address class imbalance
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # Initialize loss functions
        if use_focal_loss:
            print(f"Using Focal Loss with gamma={focal_gamma}, label_smoothing={label_smoothing}")
            self.entity_loss_fn = FocalLoss(
                alpha=entity_class_weights,
                gamma=focal_gamma,
                ignore_index=0,
                label_smoothing=label_smoothing
            )
            self.event_loss_fn = FocalLoss(
                alpha=None,  # Events are more balanced
                gamma=focal_gamma,
                ignore_index=0,
                label_smoothing=label_smoothing
            )
        else:
            self.entity_loss_fn = None
            self.event_loss_fn = None

        
    def compute_loss(self, predictions, targets, win_probs, target_win_probs=None):
        """
        Compute combined loss for build order generation.
        
        Uses Focal Loss for entity/event classification to address class imbalance.
        This prevents the model from always predicting high-frequency entities like Villager.
        
        Note: Logits are cast to float32 to ensure correct loss computation under AMP.
        """
        entity_logits, event_logits, time_preds, pred_win_probs = predictions
        entity_targets, event_targets, time_targets = targets
        
        # Classification losses - use Focal Loss if enabled, otherwise CrossEntropy
        if self.use_focal_loss and self.entity_loss_fn is not None:
            # Focal Loss handles class imbalance by down-weighting easy/common examples
            entity_loss = self.entity_loss_fn(entity_logits, entity_targets)
            event_loss = self.event_loss_fn(event_logits, event_targets)
        else:
            # Fallback to standard CrossEntropyLoss with label smoothing
            entity_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=self.label_smoothing)(
                entity_logits.view(-1, entity_logits.size(-1)).float(),
                entity_targets.view(-1)
            )
            event_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=self.label_smoothing)(
                event_logits.view(-1, event_logits.size(-1)).float(),
                event_targets.view(-1)
            )
        
        # Time prediction loss (MSE) - cast to float32 for consistency
        time_loss = nn.MSELoss()(time_preds.float(), time_targets.float())
        
        # Win probability prediction loss
        if target_win_probs is not None:
            win_loss = nn.BCEWithLogitsLoss()(pred_win_probs.float(), target_win_probs.float())
        else:
            win_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = (
            entity_loss * 1.0 +
            # event_loss * 1.0 +
            #time_loss * 0.8 + 
            win_loss * 0.3    
        )
        
        return total_loss, {
            'entity_loss': entity_loss.item(),
            'event_loss': event_loss.item(),
            'time_loss': time_loss.item(),
            'win_loss': win_loss.item() if target_win_probs is not None else 0.0,
            'total_loss': total_loss.item()
        }
    
    def train_step(self, batch, teacher_forcing_ratio=0.5):
        """Single training step. Supports AMP and gradient accumulation.
        
        Note: Always uses autoregressive generation during training (never parallel mode)
        to ensure the model learns to generate sequences without teacher forcing.
        """
        self.model.train()

        # Unpack batch
        player_civ = batch['player_civ'].to(self.device)
        enemy_civ = batch['enemy_civ'].to(self.device)
        entity_ids = batch['entity_ids'].to(self.device)
        event_ids = batch['event_ids'].to(self.device)
        times = batch['times'].to(self.device)

        # Optional: use win probability from data or reward model
        if 'labels' in batch:
            target_win_probs = batch['labels'].to(self.device).float()
        else:
            target_win_probs = None

        # Derive allowed entity mask for batch if model has civ mapping
        if hasattr(self.model, 'civ_entity_mask') and getattr(self.model, 'civ_entity_mask') is not None:
            allowed_entities_mask = self.model.civ_entity_mask[player_civ]
        else:
            allowed_entities_mask = None

        try:
            # Force autoregressive mode during training by ensuring teacher_forcing_ratio < 1.0
            # This prevents the model from relying on the parallel teacher-forcing path
            # and ensures it learns proper autoregressive generation
            training_tf_ratio = min(teacher_forcing_ratio, 0.999)  # Always < 1.0 during training
            
            # Condition Dropout: with probability 0.2, mask the win condition (set to None)
            # to force the model to learn from the sequence itself rather than just the label.
            input_win_probs = target_win_probs
            if self.model.training and target_win_probs is not None and torch.rand(1).item() < 0.2:
                 input_win_probs = None

            if self.scaler is not None:
                # AMP forward/backward with chosen dtype (float16 on CUDA, bfloat16 on ROCm)
                with torch.amp.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                    predictions = self.model(
                        player_civ=player_civ,
                        enemy_civ=enemy_civ,
                        target_sequence_length=entity_ids.size(1),
                        target_win_prob=input_win_probs,
                        teacher_forcing_ratio=training_tf_ratio,
                        ground_truth=(entity_ids, event_ids, times),
                        allowed_entities_mask=allowed_entities_mask,
                        temperature=1.0,  # Use temperature sampling during training
                        top_k=10  # Limit to top-10 choices during training
                    )
                    loss, loss_dict = self.compute_loss(
                        predictions,
                        (entity_ids, event_ids, times),
                        predictions[-1],
                        target_win_probs
                    )
                    loss = loss / self.grad_accum_steps
                    # Zero grad only at accumulation step start
                    if self._accum_step == 0:
                        self.optimizer.zero_grad()
                    # Scale and backward
                    self.scaler.scale(loss).backward()
            else:
                # Regular precision
                predictions = self.model(
                    player_civ=player_civ,
                    enemy_civ=enemy_civ,
                    target_sequence_length=entity_ids.size(1),
                    target_win_prob=input_win_probs,
                    teacher_forcing_ratio=training_tf_ratio,
                    ground_truth=(entity_ids, event_ids, times),
                    allowed_entities_mask=allowed_entities_mask,
                    temperature=1.0,  # Use temperature sampling during training
                    top_k=10  # Limit to top-10 choices during training
                )
                loss, loss_dict = self.compute_loss(
                    predictions,
                    (entity_ids, event_ids, times),
                    predictions[-1],
                    target_win_probs
                )
                loss = loss / self.grad_accum_steps
                if self._accum_step == 0:
                    self.optimizer.zero_grad()
                loss.backward()

            # Accumulation bookkeeping
            self._accum_step += 1
            if self._accum_step >= self.grad_accum_steps:
                # Unscale and clip if using scaler
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Step optimizer
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Reset accumulation
                self._accum_step = 0
        except RuntimeError as e:
            # Provide helpful OOM message and clear cache
            if 'out of memory' in str(e).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise RuntimeError("CUDA out-of-memory during training step. Consider reducing --batch_size, enabling --use_amp, or increasing --grad_accum_steps.") from e
            raise

        return loss_dict
    
    def generate_optimal_build_order(self, player_civ_id, enemy_civ_id, beam_width=5, max_length=50):
        """Generate optimal build order for given civ matchup."""
        self.model.eval()
        
        with torch.no_grad():
            player_civ = torch.tensor([player_civ_id], device=self.device)
            enemy_civ = torch.tensor([enemy_civ_id], device=self.device)
            
            sequences, win_probs = self.model.beam_search_generate(
                player_civ=player_civ,
                enemy_civ=enemy_civ,
                beam_width=beam_width,
                max_length=max_length,
                target_win_prob=None  # Generate max win probability
            )
            
            return sequences[0], win_probs[0].item()


# Utility function to load pretrained win prediction model
def load_pretrained_win_model(checkpoint_path, device='cuda'):
    """Load a pretrained win prediction model to use as reward signal."""
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocabs = checkpoint['vocabs']

    # Create win prediction model (using original architecture)
    win_model = AoETransformer(
        vocab_size_entity=len(vocabs['entity_vocab']),
        vocab_size_event=len(vocabs['event_vocab']),
        civ_vocab_size=len(vocabs['civ_vocab']),
        map_vocab_size=len(vocabs['map_vocab']),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_len=1024
    ).to(device)

    win_model.load_state_dict(checkpoint['model_state'])
    win_model.eval()

    return win_model, vocabs