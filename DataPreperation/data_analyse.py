import pandas as pd
import torch

# Load the checkpoint to get entity vocab
checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)
entity_vocab = checkpoint['entity_vocab']
inv_vocab = {v: k for k, v in entity_vocab.items()}

# Load the training data and count entity frequencies
df = pd.read_csv('transformer_input_new.csv')
df = df[df['event'] != 'DESTROY']  # Same filter as training

# Count entity frequencies
entity_counts = df['entity'].value_counts()
total = len(df)

print('Top 100 most common entities:')
for i, (entity, count) in enumerate(entity_counts.head(100).items()):
    print(f'{i+1:2d}. {entity:30s}: {count:7d} ({100*count/total:.2f}%)')

print(f'\nTotal entities: {total}')
print(f'Unique entities: {len(entity_counts)}')
print(f'Villager percentage: {100*entity_counts.get("Villager", 0)/total:.2f}%')