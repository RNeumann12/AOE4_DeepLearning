import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyGRU(nn.Module):
    """
    GRU + Asttention pooling for strategy classification
    """
    def __init__(self, num_numeric, num_events, hidden_size=128, num_layers=1, num_classes=3, dropout=0.1):
        """
        Args:
            num_numeric (int): number of numeric features per timestep
            num_events (int): number of event features (vocab size)
            hidden_size (int): hidden size of GRU
            num_layers (int): number of GRU layers
            num_classes (int): number of target strategies
            dropout (float): dropout probability
        """
        super().__init__()

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
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, numeric_seq, event_seq, mask):
        """
        Args:
            numeric_seq: [batch_size, seq_len, num_numeric]
            event_seq: [batch_size, seq_len, num_events]
            mask: [batch_size, seq_len] boolean tensor for valid timesteps
        Returns:
            logits: [batch_size, num_classes]
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
