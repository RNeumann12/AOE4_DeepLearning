import torch
import torch.nn as nn
import math

class TimePositionalEncoding(nn.Module):
    """
    Sinusoidal encoding for continuous time values.
    Allows the model to learn 'phases' of the game (Early/Mid/Late) 
    mathematically rather than trying to regress a linear value.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, times):
        """
        Args:
            times: Tensor of shape (B, L) containing integer timestamps (seconds).
        Returns:
            Tensor of shape (B, L, d_model)
        """
        # Clamp times to max_len - 1 to prevent index errors
        # Assuming times are seconds. If they are floats (0.0-1.0), scale them up to indices.
        # Here we assume integer seconds.
        times_int = times.long().clamp(0, self.pe.size(0) - 1)
        return self.pe[times_int]


class AoETransformer(nn.Module):
    def __init__(self, 
                 vocab_size_entity: int, 
                 vocab_size_event: int, 
                 civ_vocab_size: int, 
                 map_vocab_size: int, 
                 d_model: int = 128, 
                 nhead: int = 4, 
                 num_layers: int = 3, 
                 dim_feedforward: int = 256, 
                 dropout: float = 0.1, 
                 max_len: int = 1024,
                 max_time_seconds: int = 5400): # Default 90 mins max
        super().__init__()
        
        # 1. Embeddings
        self.entity_embed = nn.Embedding(vocab_size_entity, d_model)
        self.event_embed = nn.Embedding(vocab_size_event, d_model)
        self.civ_embed = nn.Embedding(civ_vocab_size, d_model)
        self.map_embed = nn.Embedding(map_vocab_size, d_model)
        
        # Positional Embeddings
        self.seq_pos_embed = nn.Embedding(max_len, d_model) # Order of events (1st, 2nd, 3rd...)
        self.time_encoding = TimePositionalEncoding(d_model, max_len=max_time_seconds) # Actual game time
        
        # [CLS] Token: A learnable parameter to aggregate sequence info
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True # Crucial for shape handling (B, L, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Classifier
        # Input: [CLS] embedding + Player Civ + Enemy Civ
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, entity_ids, event_ids, times, attention_mask, game_map, player_civ, enemy_civ):

        """
        Forward pass of the model.

        Args:
            entity_ids: Tensor of shape (B, L) containing entity IDs.
            event_ids: Tensor of shape (B, L) containing event IDs.
            times: Tensor of shape (B, L) containing time values in seconds.
            attention_mask: Tensor of shape (B, L) containing attention mask.
            game_map: Tensor of shape (B) containing map IDs.
            player_civ: Tensor of shape (B) containing player civ IDs.
            enemy_civ: Tensor of shape (B) containing enemy civ IDs.

        Returns:
            logits: Tensor of shape (B) containing predicted win probabilities.
        """
        B, L = entity_ids.size()
        
        # --- Embedding Layer ---
        # Combine Entity + Event
        x = self.entity_embed(entity_ids) + self.event_embed(event_ids)
        
        # Add Sequence Position (1st item, 2nd item...)
        seq_positions = torch.arange(0, L, device=entity_ids.device).unsqueeze(0).expand(B, L)
        x = x + self.seq_pos_embed(seq_positions)
        
        # Add Temporal Position (Time: 0s, 5s, 120s...)
        # Note: If 'times' in your dataset is scaled 0.0-1.0, multiply by max_seconds here.
        # Assuming 'times' is passed as raw seconds or close to integer steps.
        x = x + self.time_encoding(times)

        # --- Prepend [CLS] Token (Classification token) ---
        # Expand CLS to batch size: (B, 1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate CLS to the sequence -> New length L+1
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Update mask to include CLS token (always attend to CLS)
        # New mask shape: (B, L+1)
        cls_mask = torch.ones((B, 1), device=attention_mask.device)
        full_mask = torch.cat((cls_mask, attention_mask), dim=1)
        
        # Create padding mask for Transformer (True = Ignore/Pad)
        # PyTorch Transformer expects "True" for padded elements
        src_key_padding_mask = (full_mask == 0)

        # --- Transformer ---
        # Output: (B, L+1, d_model) because we used batch_first=True
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # --- Classification ---
        # Extract just the [CLS] token output (index 0)
        cls_output = encoded[:, 0, :] # (B, d_model)

        # Get Civ Embeddings
        maps = self.map_embed(game_map) # (B, d_model)
        p_civ = self.civ_embed(player_civ) # (B, d_model)
        e_civ = self.civ_embed(enemy_civ)  # (B, d_model)

        # Combine: Context (Civs) + Sequence Info (CLS)
        combined = torch.cat([cls_output, p_civ, e_civ], dim=1) # (B, 3*d_model)
        
        logits = self.classifier(combined).squeeze(-1)
        return logits
