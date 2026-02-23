import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyGRU(nn.Module):
    """
    GRU + Attention pooling for strategy classification
    Incorporates context features: civs, map, age
    """
    def __init__(self, vocab_sizes, embedding_dim=64, hidden_dim=256, num_classes=5):      
        super(StrategyGRU, self).__init__()

       # Embeddings for categorical features
       # higher embedding dim for entity (446 items in vocab)
        self.entity_emb = nn.Embedding(vocab_sizes['entity'], 64, padding_idx=0)
        self.event_emb = nn.Embedding(vocab_sizes['event'], 8, padding_idx=0)
        self.age_emb = nn.Embedding(vocab_sizes['age'], 8, padding_idx=0)
        self.type_emb = nn.Embedding(vocab_sizes['type'], 8, padding_idx=0)
        
        # Embeddings for metadata
        # 22 vocabs
        self.civ_emb = nn.Embedding(vocab_sizes['civ'], 16)
        self.enemy_civ_emb = nn.Embedding(vocab_sizes['enemy_civ'], 16)
        self.map_emb = nn.Embedding(vocab_sizes['map'], 16)

        # entities, events, age, type, time, villagers
        gru_input_dim = 64 + 8 + 8 + 8 + 2 
        self.gru = nn.GRU(gru_input_dim, hidden_dim, batch_first=True)

        # classifier
        # map, civ, civ
        fc_input_dim = hidden_dim + 3 * 16
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, sequence, mask, metadata):
        # Split sequence features
        entity = sequence[:, :, 0].long()
        event = sequence[:, :, 1].long()
        type = sequence[:, :, 2].long()
        age = sequence[:, :, 3].long()
        time = sequence[:, :, 4].unsqueeze(-1)
        villagers = sequence[:, :, 5].unsqueeze(-1)

        # Embed categorical features
        entity_emb = self.entity_emb(entity)
        event_emb = self.event_emb(event)
        type_emb = self.type_emb(type)
        age_emb = self.age_emb(age)

        # Concatenate all sequence features
        seq_input = torch.cat([entity_emb, event_emb, type_emb, age_emb, time, villagers], dim=-1)
        
        # Pack padded sequence
        lengths = mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_input, lengths, batch_first=True, enforce_sorted=False
        )
        
        # GRU
        packed_output, hidden = self.gru(packed)
        gru_out = hidden[-1]  # (batch, hidden_dim)
        
        # Embed metadata
        civ_emb = self.civ_emb(metadata[:, 0])
        enemy_civ_emb = self.enemy_civ_emb(metadata[:, 1])
        map_emb = self.map_emb(metadata[:, 2])
        
        # Combine GRU output with metadata
        combined = torch.cat([gru_out, civ_emb, enemy_civ_emb, map_emb], dim=-1)
        
        # Classifier
        output = self.fc(combined)
        return output
    

if __name__ == "__main__":
    vocab_sizes = {
        'entity': 446,
        'event': 4,
        'type': 6,
        'age': 3,
        'civ': 22,
        'enemy_civ': 22,
        'map': 22
    }
    
    model = StrategyGRU(vocab_sizes, embedding_dim=64, hidden_dim=256, num_classes=5)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Breakdown by layer:
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")
