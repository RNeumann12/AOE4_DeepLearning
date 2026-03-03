import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategyUnsupervisedEncoder(nn.Module):
    """
    GRU-based encoder that produces a fixed-size embedding for a game.

    Designed as a drop-in structural sibling to `StrategyGRU` from
    `model_v2.py` but without a supervised classification head.

    The forward() returns a dense embedding tensor (batch, embed_dim)
    which can be used for downstream unsupervised tasks (clustering,
    nearest-neighbour retrieval, etc.).
    """

    def __init__(self, 
                 vocab_sizes, 
                 embedding_dim: int = 256, 
                 seq_dropout: float = 0.1, 
                 proj_dropout: float = 0.3,
                 hidden_dim: int = 256, 
                 embed_out: int = 64,
                 gru_layers: int = 1,
                 bidirectional: bool = False,
                 normalize: bool = True
                 ):
        super(StrategyUnsupervisedEncoder, self).__init__()

        # Sequence embeddings
        self.entity_emb = nn.Embedding(vocab_sizes.get('entity', 512), 64, padding_idx=0)
        self.event_emb = nn.Embedding(vocab_sizes.get('event', 16), 8, padding_idx=0)
        self.age_emb = nn.Embedding(vocab_sizes.get('age', 8), 8, padding_idx=0)

        # Metadata embeddings (small dims)
        # self.civ_emb = nn.Embedding(vocab_sizes.get('civ', 22), 16)

        # GRU input: entity(64) + event(8) + type(8) + age(8) + time(1) + villagers(1)
        gru_input_dim = 64 + 8 + 8 + 2
        # gru_input_dim = 64 + 8 + 8 + 8 + 2
        self.gru = nn.GRU(
            input_size=gru_input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True,
            num_layers=gru_layers,
            bidirectional=bidirectional
        )

        self.gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        proj_in = self.gru_out_dim  # hidden_dim + (civ_emb + map_emb)

        self.project = nn.Sequential(
            nn.Linear(proj_in, proj_in // 2),
            nn.ReLU(),
            nn.LayerNorm(proj_in // 2),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_in // 2, embed_out)
        )

        # Dropout for sequences (variational)
        self.seq_dropout_p = seq_dropout

        # Normalize embeddings for clustering
        self.normalize = normalize

        self.entity_head = nn.Linear(embed_out, vocab_sizes.get('entity', 512))
        self.event_head  = nn.Linear(embed_out, vocab_sizes.get('event', 16))
        self.age_head    = nn.Linear(embed_out, vocab_sizes.get('age', 8))
        self.time_head   = nn.Linear(embed_out, 1)
        self.vill_head   = nn.Linear(embed_out, 1)

    def seq_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Variational dropout: same mask across timesteps."""
        if not self.training or self.seq_dropout_p == 0:
            return x
        mask = x.new_empty((x.size(0), 1, x.size(2))).bernoulli_(1 - self.seq_dropout_p)
        mask = mask / (1 - self.seq_dropout_p)
        return x * mask
    
    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: Long/Float tensor of shape (batch, seq_len, feat)
                Expected feature ordering mirrors `model_v2`:
                [entity, event, type, age, time, villagers]
            mask: Byte/Bool tensor (batch, seq_len) where 1 indicates valid tokens
            metadata: Long tensor (batch, 3) containing [civ, enemy_civ, map]

        Returns:
            embeddings: FloatTensor (batch, embed_out)
        """
        # Unpack sequence features
        entity = sequence[:, :, 0].long()
        event = sequence[:, :, 1].long()
        age = sequence[:, :, 3].long()
        time = sequence[:, :, 4].unsqueeze(-1).float()
        villagers = sequence[:, :, 5].unsqueeze(-1).float()

        # Embeddings
        entity_emb = self.entity_emb(entity)
        event_emb = self.event_emb(event)
        age_emb = self.age_emb(age)

        # Sequence input concat
        seq_input = torch.cat([entity_emb, event_emb, age_emb, time, villagers], dim=-1)
        seq_input = self.seq_dropout(seq_input)

        # Handle zero-length sequences
        lengths = mask.sum(dim=1).long().cpu()
        lengths[lengths == 0] = 1
        zero_idx = (lengths == 1).nonzero(as_tuple=False).squeeze(1)
        if zero_idx.numel() > 0:
            seq_input[zero_idx, 0, :] = 0.0

        packed = nn.utils.rnn.pack_padded_sequence(seq_input, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        if self.gru.bidirectional:
            gru_out = hidden.view(self.gru.num_layers, 2, sequence.size(0), self.gru.hidden_size)[-1]
            gru_out = torch.cat([gru_out[0], gru_out[1]], dim=-1)
        else:
            gru_out = hidden[-1]

       
        emb = self.project(gru_out)
        # emb = self.project(combined)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        # Decode
        batch_size, seq_len = sequence.shape[0], sequence.shape[1]
        emb_expanded = emb.unsqueeze(1).expand(-1, seq_len, -1)

        entity_logits = self.entity_head(emb_expanded)
        event_logits  = self.event_head(emb_expanded)
        age_logits    = self.age_head(emb_expanded)
        time_pred     = self.time_head(emb_expanded).squeeze(-1)
        vill_pred     = self.vill_head(emb_expanded).squeeze(-1)

        return emb, (entity_logits, event_logits, age_logits, time_pred, vill_pred)


if __name__ == "__main__":
    # small smoke test printing parameter counts
    vocab_sizes = {
        'entity': 446,
        'event': 16,
        'type': 6,
        'age': 3,
        'civ': 22,
        'enemy_civ': 22,
        'map': 22
    }

    model = StrategyUnsupervisedEncoder(vocab_sizes, embedding_dim=64, hidden_dim=256, embed_out=128)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, trainable: {trainable:,}")
    
    # Test forward pass
    batch_size, seq_len = 4, 50
    seq = torch.randn(batch_size, seq_len, 6)
    mask = torch.ones(batch_size, seq_len)
    meta = torch.randint(0, 20, (batch_size, 3))
    emb, recon = model(seq, mask, meta)
    print(f"Embedding shape: {emb.shape}, Reconstructed shape: {recon.shape}")
