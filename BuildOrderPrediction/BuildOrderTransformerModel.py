import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional

from WinRatePrediction.WinRateTransformerModel import AoETransformer, TimePositionalEncoding

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
        
        # 6. Win Probability Head (optional, can use pretrained model)
        self.win_prob_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Special tokens (include sequence dim so they can be expanded to (B,1,D))
        self.start_token_entity = nn.Parameter(torch.randn(1, 1, d_model))
        self.start_token_event = nn.Parameter(torch.randn(1, 1, d_model))

        # Mask sentinel value (fp16-safe) used when masking logits with AMP enabled
        # Use a large negative number that fits in float16 range (max ~ -65504). We pick -1e4.
        self.mask_value = float(-1e4)
        
    def encode_condition(self, player_civ: torch.Tensor, enemy_civ: torch.Tensor, 
                         target_win_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the condition (civ matchup + optional target win probability)
        """
        batch_size = player_civ.size(0)
        
        # Embed civilizations
        p_civ = self.civ_embed(player_civ)  # (B, d_model)
        e_civ = self.civ_embed(enemy_civ)    # (B, d_model)
        
        if target_win_prob is not None:
            # Project win probability to same dimension
            win_prob_embed = target_win_prob.unsqueeze(-1).expand(-1, self.d_model)
            condition = torch.cat([p_civ, e_civ, win_prob_embed], dim=1)
        else:
            # Use zeros for win prob dimension
            zeros = torch.zeros_like(p_civ)
            condition = torch.cat([p_civ, e_civ, zeros], dim=1)
            
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
    
    def forward(self, 
                player_civ: torch.Tensor,
                enemy_civ: torch.Tensor,
                target_sequence_length: int,
                target_win_prob: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5,
                ground_truth: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                allowed_entities_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        # Generate sequence autoregressively
        for step in range(target_sequence_length):
            # Current sequence length (including start token + generated steps)
            seq_len = step + 1
            
            # Create positional embeddings for current sequence
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            pos_emb = self.seq_pos_embed(positions)  # (B, seq_len, d_model)
            
            # Combine current embeddings with positional encoding
            decoder_input = (current_entities + current_events + pos_emb) / 3.0  # (B, seq_len, d_model)
            
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
            time_delta = torch.relu(self.time_head(last_output)).squeeze(-1)  # (B, 1)

            # Derive allowed mask if not provided
            if allowed_entities_mask is None and hasattr(self, 'civ_entity_mask') and self.civ_entity_mask is not None:
                # civ_entity_mask indexed by civ id -> (num_civs, vocab)
                allowed = self.civ_entity_mask[player_civ]
            else:
                allowed = allowed_entities_mask

            # Apply mask to entity logits if available (set disallowed logits to a very negative value)
            if allowed is not None:
                # allowed: (B, V) -> expand to (B, 1, V) to match logits
                allowed_b1v = allowed.unsqueeze(1)
                entity_logits = entity_logits.masked_fill(~allowed_b1v, self.mask_value)

            # Store logits and preds
            all_entity_logits.append(entity_logits)
            all_event_logits.append(event_logits)
            all_time_preds.append(time_delta)
            
            # Decide whether to use ground truth or predictions
            if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                next_entity = ground_truth[0][:, step:step+1]  # (B, 1)
                next_event = ground_truth[1][:, step:step+1]   # (B, 1)
                next_time = ground_truth[2][:, step:step+1].unsqueeze(-1) if step > 0 else time_delta.unsqueeze(-1)
            else:
                # Greedy sampling (can be changed to beam search/top-k during inference)
                # Argmax respects masked logits since disallowed logits are very negative
                next_entity = torch.argmax(entity_logits, dim=-1)  # (B, 1)
                next_event = torch.argmax(event_logits, dim=-1)    # (B, 1)
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

                    decoder_input = (beam['entity_embs'] + beam['event_embs'] + pos_emb) / 3.0
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
                decoder_input = (beam['entity_embs'] + beam['event_embs'] + pos_emb) / 3.0
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
    """
    def __init__(self, model, optimizer, device, use_reinforce=False, reward_model=None, use_amp: bool = False, grad_accum_steps: int = 1):
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

        
    def compute_loss(self, predictions, targets, win_probs, target_win_probs=None):
        """
        Compute combined loss for build order generation.
        """
        entity_logits, event_logits, time_preds, pred_win_probs = predictions
        entity_targets, event_targets, time_targets = targets
        
        # Classification losses for entity and event
        entity_loss = nn.CrossEntropyLoss(ignore_index=0)(
            entity_logits.view(-1, entity_logits.size(-1)),
            entity_targets.view(-1)
        )
        
        event_loss = nn.CrossEntropyLoss(ignore_index=0)(
            event_logits.view(-1, event_logits.size(-1)),
            event_targets.view(-1)
        )
        
        # Time prediction loss (MSE)
        time_loss = nn.MSELoss()(time_preds, time_targets)
        
        # Win probability prediction loss
        if target_win_probs is not None:
            win_loss = nn.BCEWithLogitsLoss()(pred_win_probs, target_win_probs)
        else:
            win_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = (
            entity_loss * 1.0 +
            event_loss * 1.0 +
            time_loss * 0.1 +  # Lower weight for time
            win_loss * 0.5     # Moderate weight for win prob
        )
        
        return total_loss, {
            'entity_loss': entity_loss.item(),
            'event_loss': event_loss.item(),
            'time_loss': time_loss.item(),
            'win_loss': win_loss.item() if target_win_probs is not None else 0.0,
            'total_loss': total_loss.item()
        }
    
    def train_step(self, batch, teacher_forcing_ratio=0.5):
        """Single training step. Supports AMP and gradient accumulation."""
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
            if self.scaler is not None:
                # AMP forward/backward with chosen dtype (float16 on CUDA, bfloat16 on ROCm)
                with torch.amp.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                    predictions = self.model(
                        player_civ=player_civ,
                        enemy_civ=enemy_civ,
                        target_sequence_length=entity_ids.size(1),
                        target_win_prob=target_win_probs,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                        ground_truth=(entity_ids, event_ids, times),
                        allowed_entities_mask=allowed_entities_mask
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
                    target_win_prob=target_win_probs,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    ground_truth=(entity_ids, event_ids, times),
                    allowed_entities_mask=allowed_entities_mask
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


# Example usage
if __name__ == "__main__":
    # Minimal example vocabs (include explicit pad id = 0 for civs)
    entity_vocab = {"villager": 1, "scout": 2, "spearman": 3, "knight": 4, "archer": 5}
    event_vocab = {"train": 1, "build": 2, "research": 3}
    civ_vocab = {"pad": 0, "English": 1, "French": 2}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create generator model and move to device
    vocab_size_entity = len(entity_vocab) + 1
    vocab_size_event = len(event_vocab) + 1
    num_civs = max(civ_vocab.values()) + 1

    generator = BuildOrderGenerator(
        vocab_size_entity=vocab_size_entity,
        vocab_size_event=vocab_size_event,
        civ_vocab_size=num_civs,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_len=100
    ).to(device)

    # Build a civ -> entity allowed mask and register it
    # Shape: (num_civs, vocab_size_entity) -- index 0 is padding civ
    civ_mask = torch.zeros((num_civs, vocab_size_entity), dtype=torch.bool)
    # English (1): allow villager(1), scout(2), spearman(3)
    civ_mask[1, 1] = True
    civ_mask[1, 2] = True
    civ_mask[1, 3] = True
    # French (2): allow villager(1), scout(2), knight(4), archer(5)
    civ_mask[2, 1] = True
    civ_mask[2, 2] = True
    civ_mask[2, 4] = True
    civ_mask[2, 5] = True

    # Register mask on the model (moves to device with the model)
    generator.set_civ_entity_mask(civ_mask.to(device))

    # Optional: Load pretrained win prediction model for reward (if available)
    try:
        win_model, vocabs = load_pretrained_win_model("best_model.pt", device=device)
    except Exception as e:
        print(f"Could not load pretrained win model: {e}")
        win_model = None

    # Create trainer
    optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    trainer = BuildOrderTrainer(generator, optimizer, device=device)

    # Create a synthetic batch and run a training step (the trainer derives allowed mask from model)
    B = 2
    L = 6
    player_civ = torch.tensor([1, 2], dtype=torch.long, device=device)
    enemy_civ = torch.tensor([2, 1], dtype=torch.long, device=device)
    entity_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    event_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    times = torch.zeros(B, L, dtype=torch.float, device=device)
    labels = torch.tensor([1, 0], dtype=torch.float, device=device)

    batch = {
        'player_civ': player_civ,
        'enemy_civ': enemy_civ,
        'entity_ids': entity_ids,
        'event_ids': event_ids,
        'times': times,
        'labels': labels
    }

    loss_dict = trainer.train_step(batch, teacher_forcing_ratio=1.0)
    print("Trainer step losses:", loss_dict)

    # Run a forward pass in inference mode using an explicit allowed mask override
    generator.eval()
    with torch.no_grad():
        # Derive allowed mask for batch from civ mask directly (same shape used by model)
        allowed_mask_batch = generator.civ_entity_mask[player_civ].to(device)
        ent_logits, ev_logits, t_preds, win_logits = generator(
            player_civ=player_civ,
            enemy_civ=enemy_civ,
            target_sequence_length=4,
            teacher_forcing_ratio=0.0,
            ground_truth=None,
            allowed_entities_mask=allowed_mask_batch
        )
    print("Example entity logits (first batch, first timestep):", ent_logits[0, 0])
    print("Win logits:", win_logits)

    # Generate an optimal build order using beam search (trainer helper)
    english_id = civ_vocab["English"]
    french_id = civ_vocab["French"]
    seq, win_prob = trainer.generate_optimal_build_order(english_id, french_id, beam_width=4, max_length=8)
    print("Generated build:", seq)
    print("Predicted win probability:", win_prob)
