import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyGRU(nn.Module):
    """
    GRU + Attention pooling for strategy classification
    Incorporates context features: civs, map, age
    """
    def __init__(self, num_numeric, num_events, hidden_size=128, num_layers=1, num_classes=3, dropout=0.1,
                 num_civs=1, num_enemy_civs=1, num_maps=1, num_ages=1):
        """
        Args:
            num_numeric (int): number of numeric features per timestep
            num_events (int): number of event features (vocab size)
            hidden_size (int): hidden size of GRU
            num_layers (int): number of GRU layers
            num_classes (int): number of target strategies
            dropout (float): dropout probability
            num_civs (int): size of player civilization vocabulary
            num_enemy_civs (int): size of enemy civilization vocabulary
            num_maps (int): size of map vocabulary
            num_ages (int): size of age/epoch vocabulary
        """
        super().__init__()

        # Embedding layers for categorical features
        self.civ_embed = nn.Embedding(num_civs, hidden_size // 4)
        self.enemy_civ_embed = nn.Embedding(num_enemy_civs, hidden_size // 4)
        self.map_embed = nn.Embedding(num_maps, hidden_size // 4)
        self.age_embed = nn.Embedding(num_ages, hidden_size // 4)

        # Linear Projections for numeric and event features
        self.numeric_proj = nn.Linear(num_numeric, hidden_size)
        self.event_proj = nn.Linear(num_events, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size*2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attn = nn.Linear(hidden_size * 2, 1)
        
        # Context features size: 4 embeddings of hidden_size//4 each
        context_size = hidden_size  # 4 * (hidden_size // 4)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 + context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, numeric_seq, event_seq, mask, time_seq=None, player_civ=None, enemy_civ=None, map_id=None, age=None):
        """
        Args:
            numeric_seq: [batch_size, seq_len, num_numeric]
            event_seq: [batch_size, seq_len, num_events]
            mask: [batch_size, seq_len] boolean tensor for valid timesteps
            time_seq: [batch_size, seq_len] time/age normalized values
            player_civ: [batch_size] player civilization indices
            enemy_civ: [batch_size] enemy civilization indices
            map_id: [batch_size] map indices
            age: [batch_size] age/epoch indices
        Returns:
            logits: [batch_size, num_classes]
            attn_weights: [batch_size, seq_len, 1]
        """

        # Project sequence features
        x_numeric = F.relu(self.numeric_proj(numeric_seq))
        x_events = F.relu(self.event_proj(event_seq))

        # Concatenate numeric and event features
        x = torch.cat([x_numeric, x_events], dim=-1)  # [batch_size, seq_len, hidden_size*2]

        gru_out, _ = self.gru(x)  # [batch_size, seq_len, hidden_size*2]

        attn_scores = self.attn(gru_out).squeeze(-1)  # [batch_size, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        pooled = torch.sum(gru_out * attn_weights, dim=1)  # [batch_size, hidden_size*2]

        # Embed context features if provided
        context_features = []
        if player_civ is not None:
            context_features.append(self.civ_embed(player_civ))
        if enemy_civ is not None:
            context_features.append(self.enemy_civ_embed(enemy_civ))
        if map_id is not None:
            context_features.append(self.map_embed(map_id))
        if age is not None:
            context_features.append(self.age_embed(age))
        
        # Concatenate context features with pooled sequence representation
        if context_features:
            context = torch.cat(context_features, dim=-1)  # [batch_size, hidden_size]
            pooled = torch.cat([pooled, context], dim=-1)  # [batch_size, hidden_size*2 + context_size]

        logits = self.classifier(pooled)  # [batch_size, num_classes]
        return logits, attn_weights  

if __name__ == "__main__":
    batch_size = 2
    seq_len = 50
    num_numeric = 15
    num_events = 100
    num_classes = 5

    model = StrategyGRU(num_numeric, num_events, hidden_size=64, num_layers=1, num_classes=num_classes)
    numeric_seq = torch.rand(batch_size, seq_len, num_numeric)
    event_seq = torch.rand(batch_size, seq_len, num_events)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    logits, attn_weights = model(numeric_seq, event_seq, mask)
    print("Logits:", logits.shape)
    print("Attention weights:", attn_weights.shape)
