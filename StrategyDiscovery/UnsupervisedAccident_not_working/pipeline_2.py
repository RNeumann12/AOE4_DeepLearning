"""
GRU Autoencoder Pipeline for Game Data Analysis
================================================
This script trains a GRU autoencoder to learn latent representations of player-game sessions,
then clusters these embeddings to discover strategy patterns.

Pipeline:
1. Load and preprocess CSV data (numerical + categorical features)
2. Group by game_id and profile_id to create sequences
3. Train GRU autoencoder with embeddings
4. Extract latent embeddings
5. Perform KMeans clustering
6. Save results to CSV

"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import wandb
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for hyperparameters and paths."""
    
    # Data paths
    data_path = "transformer_input_new.csv"
    output_dir = "outputs"
    
    # Feature definitions
    numerical_cols = ["delta_time", "wood", "stone", "food", "gold", 
                     "wood_per_min", "stone_per_min", "food_per_min", "gold_per_min"]
    categorical_cols = ["phase", "event", "entity", "player_civ", "enemy_civ", "map"]
    
    # Model hyperparameters
    embedding_dim = 16  # Dimension for categorical embeddings
    hidden_dim = 128    # GRU hidden dimension
    latent_dim = 64     # Latent vector dimension
    num_layers = 2      # Number of GRU layers
    dropout = 0.2       # Dropout rate
    
    # Training hyperparameters
    batch_size = 64
    num_epochs = 15
    learning_rate = 1e-3
    weight_decay = 1e-5
    gradient_clip = 1.0
    
    # Clustering
    n_clusters = 5
    
    # Device and optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True  # Automatic Mixed Precision
    
    # W&B
    use_wandb = True
    wandb_project = "DeepLearning-StrategieDiscovery"
    wandb_entity = "rineumann-universit-t-klagenfurt"  # Set to your W&B username if needed

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def load_and_preprocess_data(config):
    """
    Load CSV data and preprocess features.
    
    Returns:
        df: Preprocessed dataframe
        label_encoders: Dictionary of LabelEncoder objects for categorical columns
        scaler: StandardScaler for numerical features
    """
    print(f"Loading data from {config.data_path}...")
    df = pd.read_csv(config.data_path)
    
    # Verify required columns exist
    required_cols = config.numerical_cols + config.categorical_cols + ["game_id", "profile_id"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} rows, {len(df.groupby(['game_id', 'profile_id']))} unique game sessions")
    
    # Count how many unique values
    unique_enemy_civs = df['enemy_civ'].nunique()
    
    # Check for semicolons
    has_semicolon = df['enemy_civ'].astype(str).str.contains(';').sum()
    print(f"Rows with semicolons in enemy_civ: {has_semicolon} / {len(df)} ({100*has_semicolon/len(df):.1f}%)")
        
    # Encode categorical features
    # NOTE: LabelEncoders are created at runtime and not persisted to disk
    # This is intentional - encoders are recreated if script is re-run
    label_encoders = {}
    for col in config.categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Normalize numerical features
    scaler = StandardScaler()
    # df[config.numerical_cols] = scaler.fit_transform(df[config.numerical_cols])
    
    return df, label_encoders, scaler


# =============================================================================
# DATASET AND COLLATE FUNCTION
# =============================================================================

class GameSequenceDataset(Dataset):
    """
    Dataset for game sequences grouped by (game_id, profile_id).
    
    Each item is a sequence of events for one player-game session.
    """
    
    def __init__(self, df, numerical_cols, categorical_cols):
        """
        Args:
            df: Preprocessed dataframe
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Group by game_id and profile_id
        grouped = df.groupby(['game_id', 'profile_id'])
        
        self.sequences = []
        self.meta_info = []
        
        for (game_id, profile_id), group in grouped:
            # Sort by time within each group for temporal ordering
            group = group.sort_values('time')
            
            cat_features = group[categorical_cols].values 
            num_features = group[numerical_cols].values   
            
            self.sequences.append({
                'categorical': torch.LongTensor(cat_features),
                'numerical': torch.FloatTensor(num_features)
            })
            
            self.meta_info.append({
                'game_id': game_id,
                'profile_id': profile_id
            })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.meta_info[idx]


def collate_fn(batch):
    """
    Collate function to handle variable-length sequences.
    
    Pads sequences to the same length and returns batch tensors with metadata.
    
    Args:
        batch: List of (sequence, meta_info) tuples
    
    Returns:
        Dictionary containing:
            - cat_padded: Padded categorical features (batch, max_len, n_cat_features)
            - num_padded: Padded numerical features (batch, max_len, n_num_features)
            - lengths: Original sequence lengths (batch,)
            - meta: List of metadata dictionaries
    """
    sequences, meta_info = zip(*batch)
    
    # Extract categorical and numerical sequences
    cat_sequences = [s['categorical'] for s in sequences]
    num_sequences = [s['numerical'] for s in sequences]
    
    # Get sequence lengths before padding
    lengths = torch.LongTensor([len(s) for s in cat_sequences])
    
    # Pad sequences (pad_sequence expects list of tensors)
    # Shape: (max_len, batch, features) -> transpose to (batch, max_len, features)
    cat_padded = pad_sequence(cat_sequences, batch_first=True, padding_value=0)
    num_padded = pad_sequence(num_sequences, batch_first=True, padding_value=0.0)
    
    return {
        'cat_padded': cat_padded,
        'num_padded': num_padded,
        'lengths': lengths,
        'meta': list(meta_info)
    }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class GRUAutoencoder(nn.Module):
    """
    GRU-based autoencoder for sequential game data.
    
    Architecture:
    1. Embedding layer for categorical features
    2. Concatenate embedded categorical + numerical features
    3. GRU Encoder -> latent vector
    4. GRU Decoder -> reconstruction
    
    The model handles variable-length sequences using pack_padded_sequence.
    """
    
    def __init__(self, cat_vocab_sizes, num_num_features, embedding_dim, 
                 hidden_dim, latent_dim, num_layers, dropout):
        """
        Args:
            cat_vocab_sizes: List of vocabulary sizes for each categorical feature
            num_num_features: Number of numerical features
            embedding_dim: Dimension for categorical embeddings
            hidden_dim: GRU hidden dimension
            latent_dim: Latent representation dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRUAutoencoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_num_features = num_num_features
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in cat_vocab_sizes
        ])
        
        # Total input dimension: sum of embeddings + numerical features
        self.input_dim = len(cat_vocab_sizes) * embedding_dim + num_num_features
        
        # Encoder: GRU that processes sequences
        # NOTE: Dropout in GRU can cause issues on AMD GPUs (ROCm/MIOpen)
        # We use dropout=0 in GRU and apply separate dropout layer instead
        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # Always 0 - we use separate dropout layers
        )
        
        # Latent representation: compress final hidden state
        self.encoder_to_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: from latent to hidden state
        self.latent_to_decoder = nn.Linear(latent_dim, hidden_dim)
        
        # Decoder GRU
        # NOTE: Dropout in GRU can cause issues on AMD GPUs (ROCm/MIOpen)
        # We use dropout=0 in GRU and apply separate dropout layer instead
        self.decoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # Always 0 - we use separate dropout layers
        )
        
        # Output layer: reconstruct input features
        self.output_layer = nn.Linear(hidden_dim, self.input_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def embed_categorical(self, x_cat):
        """
        Embed categorical features.
        
        Args:
            x_cat: (batch, seq_len, n_cat_features)
        
        Returns:
            embedded: (batch, seq_len, n_cat_features * embedding_dim)
        """
        batch_size, seq_len, n_cat = x_cat.shape
        
        # Embed each categorical feature separately
        embedded_list = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded_list.append(emb_layer(x_cat[:, :, i]))  # (batch, seq_len, emb_dim)
        
        # Concatenate all embeddings
        embedded = torch.cat(embedded_list, dim=-1)  # (batch, seq_len, n_cat * emb_dim)
        return embedded
    
    def forward(self, x_cat, x_num, lengths):
        """
        Forward pass through autoencoder.
        
        Args:
            x_cat: Categorical features (batch, seq_len, n_cat_features)
            x_num: Numerical features (batch, seq_len, n_num_features)
            lengths: Original sequence lengths (batch,)
        
        Returns:
            reconstruction: Reconstructed input (batch, seq_len, input_dim)
            latent: Latent representation (batch, latent_dim)
        """
        batch_size = x_cat.size(0)
        
        # Embed categorical features
        x_cat_emb = self.embed_categorical(x_cat)  # (batch, seq_len, n_cat * emb_dim)
        
        # Concatenate with numerical features
        x = torch.cat([x_cat_emb, x_num], dim=-1)  # (batch, seq_len, input_dim)
        
        # Pack sequences for efficient processing
        # IMPORTANT: lengths must be on CPU for pack_padded_sequence
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # ===== ENCODER =====
        # Process through encoder GRU
        _, h_n = self.encoder(x_packed)  # h_n: (num_layers, batch, hidden_dim)
        
        # Use final layer's hidden state
        h_final = h_n[-1]  # (batch, hidden_dim)
        h_final = self.dropout(h_final)  # Apply dropout after GRU
        
        # Compress to latent representation
        latent = self.encoder_to_latent(h_final)  # (batch, latent_dim)
        latent = self.dropout(latent)
        
        # ===== DECODER =====
        # Expand latent to decoder initial hidden state
        h_decoder = self.latent_to_decoder(latent)  # (batch, hidden_dim)
        
        # Repeat for all layers
        h_decoder = h_decoder.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        
        # Decode using the same input sequence (autoencoder reconstruction)
        decoder_out, _ = self.decoder(x_packed, h_decoder)
        
        # Unpack sequences
        decoder_out, _ = pad_packed_sequence(decoder_out, batch_first=True)  # (batch, seq_len, hidden_dim)
        decoder_out = self.dropout(decoder_out)  # Apply dropout after GRU
        
        # Reconstruct input
        reconstruction = self.output_layer(decoder_out)  # (batch, seq_len, input_dim)
        
        return reconstruction, latent


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, dataloader, optimizer, scaler, config):
    """
    Train for one epoch with detailed metrics.
    
    Args:
        model: GRUAutoencoder model
        dataloader: Training dataloader
        optimizer: Optimizer
        scaler: GradScaler for AMP
        config: Configuration object
    
    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    total_cat_loss = 0
    total_num_loss = 0
    total_samples = 0
    latent_norms = []
    grad_norms = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        cat_padded = batch['cat_padded'].to(config.device)
        num_padded = batch['num_padded'].to(config.device)
        lengths = batch['lengths']
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if config.use_amp:
            with torch.cuda.amp.autocast():
                reconstruction, latent = model(cat_padded, num_padded, lengths)
                
                # Compute target: concatenate embedded categorical + numerical
                with torch.no_grad():
                    cat_emb = model.embed_categorical(cat_padded)
                    target = torch.cat([cat_emb, num_padded], dim=-1)
                
                # Create mask for valid positions
                mask = torch.arange(cat_padded.size(1), device=config.device).unsqueeze(0) < lengths.unsqueeze(1).to(config.device)
                mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
                
                # Total MSE loss
                loss = ((reconstruction - target) ** 2 * mask).sum() / mask.sum()
                
                # Separate categorical and numerical losses for monitoring
                cat_dim = cat_emb.size(-1)
                cat_loss = ((reconstruction[:, :, :cat_dim] - target[:, :, :cat_dim]) ** 2 * mask).sum() / mask.sum()
                num_loss = ((reconstruction[:, :, cat_dim:] - target[:, :, cat_dim:]) ** 2 * mask).sum() / mask.sum()
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Track gradient norm
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            grad_norms.append(total_norm.item())
            
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstruction, latent = model(cat_padded, num_padded, lengths)
            
            # Compute target
            with torch.no_grad():
                cat_emb = model.embed_categorical(cat_padded)
                target = torch.cat([cat_emb, num_padded], dim=-1)
            
            # MSE loss with mask
            mask = torch.arange(cat_padded.size(1), device=config.device).unsqueeze(0) < lengths.unsqueeze(1).to(config.device)
            mask = mask.unsqueeze(-1).float()
            loss = ((reconstruction - target) ** 2 * mask).sum() / mask.sum()
            
            # Separate losses
            cat_dim = cat_emb.size(-1)
            cat_loss = ((reconstruction[:, :, :cat_dim] - target[:, :, :cat_dim]) ** 2 * mask).sum() / mask.sum()
            num_loss = ((reconstruction[:, :, cat_dim:] - target[:, :, cat_dim:]) ** 2 * mask).sum() / mask.sum()
            
            # Backward pass
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            grad_norms.append(total_norm.item())
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(lengths)
        total_cat_loss += cat_loss.item() * len(lengths)
        total_num_loss += num_loss.item() * len(lengths)
        total_samples += len(lengths)
        
        # Track latent space statistics
        with torch.no_grad():
            latent_norms.append(torch.norm(latent, dim=1).mean().item())
    
    # Compute average metrics
    metrics = {
        'train_loss': total_loss / total_samples,
        'train_cat_loss': total_cat_loss / total_samples,
        'train_num_loss': total_num_loss / total_samples,
        'train_latent_norm': np.mean(latent_norms),
        'train_grad_norm': np.mean(grad_norms),
        'train_grad_norm_std': np.std(grad_norms)
    }
    
    return metrics


@torch.no_grad()
def evaluate_model(model, dataloader, config):
    """
    Evaluate model with comprehensive metrics.
    
    Args:
        model: GRUAutoencoder model
        dataloader: Dataloader
        config: Configuration object
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    total_cat_loss = 0
    total_num_loss = 0
    total_samples = 0
    latent_norms = []
    all_latents = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        cat_padded = batch['cat_padded'].to(config.device)
        num_padded = batch['num_padded'].to(config.device)
        lengths = batch['lengths']
        
        reconstruction, latent = model(cat_padded, num_padded, lengths)
        
        # Compute target
        cat_emb = model.embed_categorical(cat_padded)
        target = torch.cat([cat_emb, num_padded], dim=-1)
        
        # Create mask
        mask = torch.arange(cat_padded.size(1), device=config.device).unsqueeze(0) < lengths.unsqueeze(1).to(config.device)
        mask = mask.unsqueeze(-1).float()
        
        # Total loss
        loss = ((reconstruction - target) ** 2 * mask).sum() / mask.sum()
        
        # Separate losses
        cat_dim = cat_emb.size(-1)
        cat_loss = ((reconstruction[:, :, :cat_dim] - target[:, :, :cat_dim]) ** 2 * mask).sum() / mask.sum()
        num_loss = ((reconstruction[:, :, cat_dim:] - target[:, :, cat_dim:]) ** 2 * mask).sum() / mask.sum()
        
        # Track metrics
        total_loss += loss.item() * len(lengths)
        total_cat_loss += cat_loss.item() * len(lengths)
        total_num_loss += num_loss.item() * len(lengths)
        total_samples += len(lengths)
        
        # Track latent statistics
        latent_norms.append(torch.norm(latent, dim=1).mean().item())
        all_latents.append(latent.cpu().numpy())
    
    # Compute latent space statistics
    all_latents = np.vstack(all_latents)
    latent_mean = np.mean(all_latents, axis=0)
    latent_std = np.std(all_latents, axis=0)
    
    metrics = {
        'val_loss': total_loss / total_samples,
        'val_cat_loss': total_cat_loss / total_samples,
        'val_num_loss': total_num_loss / total_samples,
        'val_latent_norm': np.mean(latent_norms),
        'val_latent_mean': np.mean(latent_mean),
        'val_latent_std': np.mean(latent_std)
    }
    
    return metrics


@torch.no_grad()
def extract_latent_embeddings(model, dataloader, config):
    """
    Extract latent embeddings for all sequences.
    
    Args:
        model: Trained GRUAutoencoder
        dataloader: Dataloader
        config: Configuration object
    
    Returns:
        embeddings: numpy array of latent vectors (n_samples, latent_dim)
        meta_info: List of metadata dictionaries
    """
    model.eval()
    
    all_latents = []
    all_meta = []
    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        cat_padded = batch['cat_padded'].to(config.device)
        num_padded = batch['num_padded'].to(config.device)
        lengths = batch['lengths']
        meta = batch['meta']
        
        _, latent = model(cat_padded, num_padded, lengths)
        
        all_latents.append(latent.cpu().numpy())
        all_meta.extend(meta)
    
    embeddings = np.vstack(all_latents)
    return embeddings, all_meta


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training and analysis pipeline."""
    
    config = Config()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # =============================================================================
    # STEP 1: Load and preprocess data
    # Sizes for categorical vocabularies
    # =============================================================================
    df, label_encoders, scaler = load_and_preprocess_data(config)
    
    cat_vocab_sizes = [len(label_encoders[col].classes_) for col in config.categorical_cols]
        
    # =============================================================================
    # STEP 2: Create dataset and dataloader
    # Split into train and validation sets (80/20 split)
    # =============================================================================
    
    dataset = GameSequenceDataset(df, config.numerical_cols, config.categorical_cols)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for Windows compatibility, increase for Linux/Mac
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    # Full dataloader for final embedding extraction
    full_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    print(f"Created dataset with {len(dataset)} sequences")
    print(f"  Training: {len(train_dataset)} sequences (80%)")
    print(f"  Validation: {len(val_dataset)} sequences (20%)")

    # =============================================================================
    # STEP 3: Initialize model
    # =============================================================================
    
    model = GRUAutoencoder(
        cat_vocab_sizes=cat_vocab_sizes,
        num_num_features=len(config.numerical_cols),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    print(f"\nModel architecture:\n{model}")
    print(f"Using device: {config.device}")
    
    # =============================================================================
    # STEP 4: Initialize optimizer and AMP
    # =============================================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # IMPORTANT: GradScaler for AMP - only use with CUDA
    # If using CPU, AMP is disabled automatically
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and config.device == "cuda")

#     # =============================================================================
#     # STEP 5: Initialize W&B
#     # =============================================================================
    
#     if config.use_wandb:
#         wandb.init(
#             project=config.wandb_project,
#             entity=config.wandb_entity,
#             config={
#                 # Only log hyperparameters, NOT runtime-dependent values
#                 "embedding_dim": config.embedding_dim,
#                 "hidden_dim": config.hidden_dim,
#                 "latent_dim": config.latent_dim,
#                 "num_layers": config.num_layers,
#                 "dropout": config.dropout,
#                 "batch_size": config.batch_size,
#                 "num_epochs": config.num_epochs,
#                 "learning_rate": config.learning_rate,
#                 "weight_decay": config.weight_decay,
#                 "gradient_clip": config.gradient_clip,
#                 "n_clusters": config.n_clusters,
#                 "use_amp": config.use_amp,
#                 "device": config.device
#             }
#         )
#         # wandb.watch(model, log="all", log_freq=100)
#         wandb.watch(model, log="gradients", log_freq=500)
    # =============================================================================
    # STEP 6: Training loop
    # =============================================================================
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, scaler, config)
        
        # Evaluate
        val_metrics = evaluate_model(model, val_dataloader, config)
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.6f} | Val Loss: {val_metrics['val_loss']:.6f}")
        print(f"  Train (Cat/Num): {train_metrics['train_cat_loss']:.6f} / {train_metrics['train_num_loss']:.6f}")
        print(f"  Val (Cat/Num): {val_metrics['val_cat_loss']:.6f} / {val_metrics['val_num_loss']:.6f}")
        print(f"  Latent Norm: {train_metrics['train_latent_norm']:.4f} | Grad Norm: {train_metrics['train_grad_norm']:.4f}")

#         # Log to W&B
#         if config.use_wandb:
#             wandb.log(epoch_metrics)
        
#         # Save best model based on validation loss
#         if val_metrics['val_loss'] < best_val_loss:
#             best_val_loss = val_metrics['val_loss']
#             patience_counter = 0
#             torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
#             print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"\nEarly stopping triggered after {epoch + 1} epochs")
#                 break
        
#         print()
    
    print("Training complete!")
    
#     # =============================================================================
#     # STEP 7: Extract latent embeddings
#     # =============================================================================
    
#     print("\n" + "="*50)
#     print("Extracting latent embeddings...")
#     print("="*50 + "\n")
    
#     # Load best model
#     model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
    
#     embeddings, meta_info = extract_latent_embeddings(model, full_dataloader, config)
    
#     print(f"Extracted {len(embeddings)} latent embeddings of dimension {embeddings.shape[1]}")
    
#     # Save embeddings to CSV
#     embeddings_df = pd.DataFrame(embeddings, columns=[f"latent_{i}" for i in range(config.latent_dim)])
#     embeddings_df['game_id'] = [m['game_id'] for m in meta_info]
#     embeddings_df['profile_id'] = [m['profile_id'] for m in meta_info]
    
#     embeddings_path = os.path.join(config.output_dir, "latent_embeddings.csv")
#     embeddings_df.to_csv(embeddings_path, index=False)
#     print(f"Saved latent embeddings to {embeddings_path}")
    
#     # =============================================================================
#     # STEP 8: Clustering
#     # =============================================================================
    
#     print("\n" + "="*50)
#     print(f"Performing KMeans clustering (k={config.n_clusters})...")
#     print("="*50 + "\n")
    
#     kmeans = KMeans(n_clusters=config.n_clusters, random_state=42, n_init=10)

#     cluster_labels = kmeans.fit_predict(embeddings)
    
#     # Add cluster assignments
#     embeddings_df['cluster'] = cluster_labels
    
#     # Save clustering results
#     clusters_path = os.path.join(config.output_dir, "cluster_assignments.csv")
#     # cluster_df = embeddings_df[['game_id', 'profile_id', 'cluster']]

    
#     latent_columns = [f"latent_{i}" for i in range(config.latent_dim)]

#     cluster_df = add_kmeans_decision_info(
#         embeddings_df,
#         kmeans,
#         latent_columns
#     )

#     cluster_df.to_csv(clusters_path, index=False)
#     print(f"Saved cluster assignments to {clusters_path}")
    
#     # Print cluster statistics
#     print("\nCluster distribution:")
#     cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
#     for cluster, count in cluster_counts.items():
#         pct = 100 * count / len(cluster_labels)
#         print(f"  Cluster {cluster}: {count} games ({pct:.1f}%)")
    
#     # Compute clustering metrics
#     from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
#     silhouette = silhouette_score(embeddings, cluster_labels)
#     calinski = calinski_harabasz_score(embeddings, cluster_labels)
#     davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    
#     print(f"\nClustering Quality Metrics:")
#     print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
#     print(f"  Calinski-Harabasz Score: {calinski:.2f} (higher is better)")
#     print(f"  Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    
#     clustering_metrics = {
#         'silhouette_score': silhouette,
#         'calinski_harabasz_score': calinski,
#         'davies_bouldin_score': davies_bouldin,
#         'n_clusters': config.n_clusters
#     }
    
#     # Add cluster distribution
#     for cluster, count in cluster_counts.items():
#         clustering_metrics[f'cluster_{cluster}_count'] = int(count)
#         clustering_metrics[f'cluster_{cluster}_pct'] = float(count / len(cluster_labels))
    
#     if config.use_wandb:
#         wandb.log(clustering_metrics)
#         wandb.log({"cluster_distribution": wandb.Histogram(cluster_labels)})
    
#     # =============================================================================
#     # STEP 9: Run Cluster Analysis and Log to W&B
#     # =============================================================================
    
#     print("\n" + "="*50)
#     print("Running cluster analysis...")
#     print("="*50 + "\n")
    
#     # Import and run cluster analysis
#     from cluster_analysis import run_analysis_and_log_to_wandb
    
#     analysis_results = run_analysis_and_log_to_wandb(
#         original_data_path=config.data_path,
#         cluster_assignments_path=clusters_path,
#         numerical_cols=config.numerical_cols,
#         categorical_cols=config.categorical_cols,
#         output_dir="cluster_analysis",
#         use_wandb=config.use_wandb
#     )
    
#     # log_cluster_scatter_wandb(
#     #     final_df,
#     #     latent_columns
#     # )


#     if config.use_wandb:
#         wandb.finish()
    
#     print("\n" + "="*50)
#     print("Pipeline complete!")
#     print("="*50)
#     print(f"\nOutputs saved to: {config.output_dir}/")
#     print(f"  - best_model.pt: Trained model weights")
#     print(f"  - latent_embeddings.csv: Latent vectors with metadata")
#     print(f"  - cluster_assignments.csv: Cluster labels for each session")
#     print(f"\nCluster analysis saved to: cluster_analysis/")
#     print(f"  - Visualizations (PNG files)")
#     print(f"  - Statistical profiles (CSV files)")
#     print(f"  - All metrics logged to W&B")


# # =============================================================================
# # POTENTIAL RUNTIME ISSUES AND RECOMMENDATIONS
# # =============================================================================
# """
# POTENTIAL RUNTIME ISSUES TO BE AWARE OF:

# 1. **GPU Memory**
#    - Issue: Large batch sizes or long sequences may cause OOM errors
#    - Solution: Reduce batch_size in Config, or use gradient accumulation
#    - Monitor: Watch GPU memory usage with nvidia-smi

# 2. **AMP (Automatic Mixed Precision)**
#    - Issue: AMP only works with CUDA. Script automatically disables for CPU
#    - Note: Some operations don't support FP16, but PyTorch handles this automatically
#    - Recommendation: Keep use_amp=True for GPU, it's ignored on CPU

# 3. **Variable-Length Sequences**
#    - Issue: pack_padded_sequence requires lengths on CPU
#    - Solution: Script already handles this with lengths.cpu()
#    - Note: enforce_sorted=False allows unsorted batch (slight performance cost)

# 4. **Label Encoding**
#    - Issue: New categories in test data will cause errors
#    - Solution: This script is for training only. For inference, save encoders with pickle
#    - Note: Encoders are recreated each run - this is intentional for this pipeline

# 5. **W&B Integration**
#    - Issue: Runtime values like cat_vocab_sizes shouldn't be in wandb.config
#    - Solution: Script only logs hyperparameters, not data-dependent values
#    - Note: Set use_wandb=False if W&B is not installed or you don't want logging

# 6. **DataLoader num_workers**
#    - Issue: num_workers > 0 can cause issues on Windows
#    - Solution: Set to 0 for Windows, increase (e.g., 4) for Linux/Mac for faster loading
#    - Current: Set to 0 for maximum compatibility

# 7. **Gradient Clipping**
#    - Issue: Essential for RNNs to prevent exploding gradients
#    - Note: Applied before optimizer step in both AMP and non-AMP paths

# 8. **Sequence Sorting**
#    - Issue: Data is sorted by time within each group, but not required to be sorted across groups
#    - Note: GRU handles temporal order based on the sequence itself

# 9. **CSV Column Requirements**
#    - Issue: Missing columns will cause immediate error
#    - Solution: Script validates all required columns exist before processing
#    - Check: Ensure game_id and profile_id exist in your CSV

# 10. **Output Directory**
#     - Issue: Permission errors if output_dir is not writable
#     - Solution: Script creates directory with os.makedirs(exist_ok=True)
#     - Note: Change output_dir in Config if needed
# """

# from sklearn.preprocessing import StandardScaler
# def add_kmeans_decision_info(
#     latent_df: pd.DataFrame,
#     kmeans: KMeans,
#     latent_columns: list[str],
# ) -> pd.DataFrame:
#     """
#     Adds distance-to-centroid information explaining
#     why each sample was assigned to its cluster.
#     """

#     scaler = StandardScaler()
#     latent_scaled = scaler.fit_transform(latent_df[latent_columns])

#     # Distances to each centroid: shape (N, K)
#     distances = kmeans.transform(latent_scaled)

#     # Add per-cluster distances
#     for i in range(distances.shape[1]):
#         latent_df[f"dist_cluster_{i}"] = distances[:, i]

#     # Distance to assigned cluster
#     latent_df["assigned_cluster_distance"] = distances[
#         range(len(distances)),
#         latent_df["cluster"].values
#     ]

#     # Optional: soft confidence score (higher = more confident)
#     latent_df["cluster_confidence"] = (
#         1.0 / (latent_df["assigned_cluster_distance"] + 1e-8)
#     )

#     return latent_df

# from sklearn.decomposition import PCA

# def log_cluster_scatter_wandb(
#     latent_df: pd.DataFrame,
#     latent_columns: list[str],
#     step: int | None = None,
# ):
#     """
#     Logs a 2D PCA projection of latent embeddings to W&B,
#     colored by cluster.
#     """

#     X = latent_df[latent_columns].values
#     clusters = latent_df["cluster"].values

#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X)

#     plot_df = pd.DataFrame({
#         "pca_1": X_2d[:, 0],
#         "pca_2": X_2d[:, 1],
#         "cluster": clusters,
#     })

#     wandb.log({
#         "cluster_scatter": wandb.plot.scatter(
#             plot_df,
#             x="pca_1",
#             y="pca_2",
#             color="cluster",
#             title="Latent Space Clusters (PCA)"
#         )
#     }, step=step)


if __name__ == "__main__":
    main()
