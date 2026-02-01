import torch
from torch.utils.data import Dataset

class AoE4Dataset(Dataset):
    def __init__(self, X_seq, X_mask, X_meta, y):
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_mask = torch.FloatTensor(X_mask)
        self.X_meta = torch.LongTensor(X_meta)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.X_seq[idx],
            'mask': self.X_mask[idx],
            'metadata': self.X_meta[idx],
            'label': self.y[idx]
        }

if __name__ == "__main__":
    # Test the dataset
    dataset = AoE4Dataset('transformer_input_test_v2.csv', seq_len=50)
    print(f"\n✓ Dataset created successfully!")
   
