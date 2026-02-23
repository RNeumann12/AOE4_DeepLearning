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

    def __init__(self, vocab_sizes, embedding_dim: int = 64, hidden_dim: int = 256, embed_out: int = 128,
                 seq_dropout: float = 0.1, proj_dropout: float = 0.3):
        super(StrategyUnsupervisedEncoder, self).__init__()

        # Sequence embeddings
        self.entity_emb = nn.Embedding(vocab_sizes.get('entity', 512), 64, padding_idx=0)
        self.event_emb = nn.Embedding(vocab_sizes.get('event', 16), 8, padding_idx=0)
        self.age_emb = nn.Embedding(vocab_sizes.get('age', 8), 8, padding_idx=0)
        self.type_emb = nn.Embedding(vocab_sizes.get('type', 8), 8, padding_idx=0)

        # Metadata embeddings (small dims)
        self.civ_emb = nn.Embedding(vocab_sizes.get('civ', 22), 16)
        self.enemy_civ_emb = nn.Embedding(vocab_sizes.get('enemy_civ', 22), 16)
        self.map_emb = nn.Embedding(vocab_sizes.get('map', 22), 16)

        # GRU input: entity(64) + event(8) + type(8) + age(8) + time(1) + villagers(1)
        gru_input_dim = 64 + 8 + 8 + 8 + 2
        self.gru = nn.GRU(gru_input_dim, hidden_dim, batch_first=True)

        # Projection head: compress GRU output + metadata into embedding
        proj_in = hidden_dim + 3 * 16
        self.project = nn.Sequential(
            nn.Linear(proj_in, proj_in // 2),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_in // 2, embed_out),
        )

        # dropout on sequence inputs (applied before packing) and projection
        self.seq_dropout = nn.Dropout(seq_dropout)

        # optional normalization (useful for clustering / similarity)
        self.normalize = True

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
        type_ = sequence[:, :, 2].long()
        age = sequence[:, :, 3].long()
        time = sequence[:, :, 4].unsqueeze(-1).float()
        villagers = sequence[:, :, 5].unsqueeze(-1).float()

        # Embeddings
        entity_emb = self.entity_emb(entity)
        event_emb = self.event_emb(event)
        type_emb = self.type_emb(type_)
        age_emb = self.age_emb(age)

        # Sequence input concat
        seq_input = torch.cat([entity_emb, event_emb, type_emb, age_emb, time, villagers], dim=-1)
        # apply small dropout to sequence inputs to regularize encoder
        seq_input = self.seq_dropout(seq_input)

        # Pack using lengths from mask
        lengths = mask.sum(dim=1).long().cpu()
        # pack_padded_sequence errors if any length == 0 — replace zeros with 1
        if (lengths == 0).any():
            zero_idx = (lengths == 0).nonzero(as_tuple=False).squeeze(1)
            lengths[zero_idx] = 1
            # Ensure those sequences have a valid (zero) timestep so GRU can process
            if seq_input.device.type != 'cpu':
                zidx = zero_idx.to(seq_input.device)
            else:
                zidx = zero_idx
            if zidx.numel() > 0:
                seq_input[zidx, 0, :] = 0.0

        packed = nn.utils.rnn.pack_padded_sequence(seq_input, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        gru_out = hidden[-1]  # (batch, hidden_dim)

        # Metadata embeddings
        civ_emb = self.civ_emb(metadata[:, 0])
        enemy_emb = self.enemy_civ_emb(metadata[:, 1])
        map_emb = self.map_emb(metadata[:, 2])

        combined = torch.cat([gru_out, civ_emb, enemy_emb, map_emb], dim=-1)

        emb = self.project(combined)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb


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
