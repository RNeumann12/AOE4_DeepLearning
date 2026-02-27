"""Train `StrategyUnsupervisedEncoder` with autoencoder MSE reconstruction loss,
then cluster embeddings and save assignments + t-SNE plot.

Autoencoder approach:
- Encoder: sequence -> embedding (B, embed_out)
- Decoder: embedding -> reconstructed sequence (B, S, 6)
- Loss: MSE reconstruction between input and reconstructed sequences
- Single embedding per sample, no pairs needed
- Clear interpretable loss signal
"""
from __future__ import annotations
import os
import argparse

import numpy as np
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model_unsupervised import StrategyUnsupervisedEncoder


def load_npz(path: str = "aoe4_dataset.npz"):
    """Load dataset from NPZ file."""
    data = np.load(path, allow_pickle=True)
    X_seq = data["X_seq"]
    X_mask = data["X_mask"]
    X_meta = data["X_meta"]
    # vocabs
    entity_vocab = data["entity_vocab"].item()
    event_vocab = data["event_vocab"].item()
    type_vocab = data["type_vocab"].item()
    age_vocab = data["age_vocab"].item()
    civ_vocab = data["civ_vocab"].item()
    enemy_civ_vocab = data["enemy_civ_vocab"].item()
    map_vocab = data["map_vocab"].item()

    vocabs = {
        "entity": len(entity_vocab) + 1,
        "event": len(event_vocab) + 1,
        "type": len(type_vocab) + 1,
        "age": len(age_vocab) + 1,
        "civ": len(civ_vocab) + 1,
        "enemy_civ": len(enemy_civ_vocab) + 1,
        "map": len(map_vocab) + 1,
    }

    return X_seq, X_mask, X_meta, vocabs


def collate_from_arrays(X_seq, X_mask, X_meta, indices):
    seq = torch.from_numpy(X_seq[indices]).float()
    mask = torch.from_numpy(X_mask[indices]).float()
    meta = torch.from_numpy(X_meta[indices]).long()
    return seq, mask, meta


def train(args):
    """Train autoencoder with reconstruction loss."""
    # device selection
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    X_seq, X_mask, X_meta, vocabs = load_npz(args.npz)

    # simple train/val split by index
    n = X_seq.shape[0]
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(n * (1 - args.val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Batch generator
    def batch_generator(indices):
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            seq, mask, meta = collate_from_arrays(X_seq, X_mask, X_meta, batch_idx)
            yield seq, mask, meta

    # Model and optimizer
    model = StrategyUnsupervisedEncoder(
        vocabs, 
        embed_out=args.embed_dim, 
        seq_dropout=args.seq_dropout, 
        proj_dropout=args.proj_dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Reconstruction loss: MSE between input and reconstructed sequence
    criterion = nn.MSELoss()

    loss_history = []
    silhouette_history = []
    epoch_metrics = []
    best_sil = -999.0
    best_epoch = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        num_batches = int(np.ceil(len(train_idx) / args.batch_size))
        pbar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}", leave=False)
        
        for seq_b, mask_b, meta_b in batch_generator(train_idx):
            seq_b = seq_b.to(device)
            mask_b = mask_b.to(device)
            meta_b = meta_b.to(device)

            # Forward: encoder produces embedding, decoder reconstructs sequence
            embedding, reconstructed = model(seq_b, mask_b, meta_b)

            # MSE reconstruction loss: compare reconstructed vs original
            # Apply mask to only compute loss on valid tokens
            loss = criterion(reconstructed * mask_b.unsqueeze(-1), seq_b * mask_b.unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        avg_loss = total_loss / max(1, steps)
        loss_history.append(avg_loss)
        
        # Save loss history
        os.makedirs(args.out_dir, exist_ok=True)
        np.save(os.path.join(args.out_dir, "loss_history.npy"), np.array(loss_history))
        print(f"Epoch {epoch+1}/{args.epochs} - avg_loss: {avg_loss:.4f}")

        # Silhouette monitoring on validation subset
        try:
            model.eval()
            val_n = len(val_idx)
            sample_n = min(val_n, args.silhouette_sample)
            
            if sample_n > 0:
                sel = np.random.choice(val_idx, sample_n, replace=False)
                with torch.no_grad():
                    seq_v = torch.from_numpy(X_seq[sel]).float().to(device)
                    mask_v = torch.from_numpy(X_mask[sel]).to(device)
                    meta_v = torch.from_numpy(X_meta[sel]).long().to(device)
                    emb_v, _ = model(seq_v, mask_v, meta_v)
                    emb_v = emb_v.cpu().numpy()

                # Optional dimensionality reduction for stability
                if emb_v.shape[0] > 1000 and emb_v.shape[1] > 10:
                    pca = PCA(n_components=min(50, emb_v.shape[1]))
                    emb_red = pca.fit_transform(emb_v)
                else:
                    emb_red = emb_v

                # Repeated KMeans for stable silhouette estimation
                sils = []
                cluster_sizes_runs = []
                for r in range(args.silhouette_repeats):
                    kmr = KMeans(n_clusters=args.n_clusters, random_state=r, n_init=10)
                    labr = kmr.fit_predict(emb_red)
                    if len(set(labr)) > 1:
                        silr = silhouette_score(emb_red, labr)
                    else:
                        silr = float('nan')
                    sils.append(silr)
                    
                    # Cluster sizes
                    counts = np.array([(labr == k).sum() for k in range(args.n_clusters)])
                    cluster_sizes_runs.append(counts.tolist())

                sil_mean = float(np.nanmean(sils))
                sil_std = float(np.nanstd(sils))
                silhouette_history.append(sil_mean)
                np.save(os.path.join(args.out_dir, "silhouette_history.npy"), np.array(silhouette_history))

                # Embedding variance
                emb_var = float(np.mean(np.std(emb_v, axis=0)))

                # Cluster sizes stats
                cluster_sizes_arr = np.array(cluster_sizes_runs)
                cluster_sizes_mean = cluster_sizes_arr.mean(axis=0).tolist()
                cluster_sizes_std = cluster_sizes_arr.std(axis=0).tolist()

                # Record epoch metrics
                em = {
                    'epoch': epoch + 1,
                    'silhouette_mean': sil_mean,
                    'silhouette_std': sil_std,
                    'cluster_sizes_mean': cluster_sizes_mean,
                    'cluster_sizes_std': cluster_sizes_std,
                    'embedding_var_mean': emb_var,
                    'sample_n': int(sample_n),
                    'avg_loss': float(avg_loss)
                }
                epoch_metrics.append(em)
                
                with open(os.path.join(args.out_dir, 'epoch_metrics.jsonl'), 'a') as fh:
                    fh.write(json.dumps(em) + "\n")

                print(f"  silhouette (sample {sample_n}): mean={sil_mean:.4f} std={sil_std:.4f}")

                # Early stopping based on silhouette improvement
                if not np.isnan(sil_mean) and sil_mean > best_sil:
                    best_sil = sil_mean
                    best_epoch = epoch + 1
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model_by_silhouette.pt'))
                    print(f"  -> New best silhouette {best_sil:.4f} (epoch {best_epoch}), model saved")
                else:
                    patience_counter += 1
                    print(f"  patience {patience_counter}/{args.patience}")

                if patience_counter >= args.patience:
                    print(f"Early stopping triggered (no improvement in {args.patience} epochs)")
                    break

        except Exception as e:
            print(f"Silhouette monitoring skipped due to error: {e}")

    # Compute embeddings for all games
    print("\nComputing final embeddings...")
    model.eval()
    with torch.no_grad():
        all_emb = []
        for i in range(0, n, args.batch_size):
            seq = torch.from_numpy(X_seq[i : i + args.batch_size]).float().to(device)
            mask = torch.from_numpy(X_mask[i : i + args.batch_size]).to(device)
            meta = torch.from_numpy(X_meta[i : i + args.batch_size]).long().to(device)
            emb, _ = model(seq, mask, meta)
            all_emb.append(emb.cpu().numpy())

    all_emb = np.vstack(all_emb)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save embeddings
    np.save(os.path.join(args.out_dir, 'embeddings.npy'), all_emb)
    print(f"Saved embeddings: {all_emb.shape}")

    # Final clustering with repeated KMeans
    print(f"\nFinal clustering with K={args.n_clusters}...")
    sils_final = []
    labels_runs = []
    for r in range(args.cluster_repeats):
        kmf = KMeans(n_clusters=args.n_clusters, random_state=r, n_init=10)
        labf = kmf.fit_predict(all_emb)
        labels_runs.append(labf)
        if len(set(labf)) > 1:
            sils_final.append(silhouette_score(all_emb, labf))
        else:
            sils_final.append(float('nan'))

    sil_final_mean = float(np.nanmean(sils_final))
    sil_final_std = float(np.nanstd(sils_final))
    
    # Save cluster assignments and summary
    np.savetxt(os.path.join(args.out_dir, "cluster_labels_kmeans.csv"), labels_runs[-1], fmt="%d")
    with open(os.path.join(args.out_dir, 'kmeans_silhouette_summary.json'), 'w') as fh:
        json.dump({'mean': sil_final_mean, 'std': sil_final_std, 'repeats': args.cluster_repeats}, fh)
    
    print(f"Final silhouette score: {sil_final_mean:.4f} ± {sil_final_std:.4f}")

    # t-SNE visualization
    print("Generating t-SNE visualization...")
    try:
        tsne = TSNE(n_components=2, random_state=0)
        proj = tsne.fit_transform(all_emb)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels_runs[-1], cmap="tab10", s=6)
        plt.colorbar(sc, label="cluster")
        plt.title("KMeans clusters (t-SNE)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "clusters_kmeans_tsne.png"), dpi=150)
        plt.close()
        print("Saved t-SNE plot")
    except Exception as e:
        print(f"t-SNE plot failed: {e}")

    # Loss and silhouette history plot
    print("Generating loss and silhouette history plot...")
    try:
        plt.figure(figsize=(8, 4))
        epochs = np.arange(1, len(loss_history) + 1)
        plt.plot(epochs, loss_history, label='reconstruction loss')
        if len(silhouette_history) > 0:
            plt.plot(epochs[: len(silhouette_history)], silhouette_history, label='silhouette')
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.legend()
        plt.title('Reconstruction Loss and Silhouette Score over Epochs')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'loss_silhouette.png'))
        plt.close()
    except Exception as e:
        print(f"Loss/silhouette plot failed: {e}")

    print(f"\nTraining complete. All outputs saved to {args.out_dir}")


def cli():
    parser = argparse.ArgumentParser(description="Train unsupervised autoencoder for AoE4 clustering")
    parser.add_argument("--npz", type=str, default="aoe4_dataset.npz", help="Path to NPZ dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--out_dir", type=str, default="outputs/unsup", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"], help="Device")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on silhouette")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--silhouette_repeats", type=int, default=5, help="KMeans repeats for validation silhouette")
    parser.add_argument("--cluster_repeats", type=int, default=10, help="KMeans repeats for final clustering")
    parser.add_argument("--seq_dropout", type=float, default=0.1, help="Sequence dropout rate")
    parser.add_argument("--proj_dropout", type=float, default=0.3, help="Projection dropout rate")
    parser.add_argument("--silhouette_sample", type=int, default=1000, help="Validation samples for silhouette (0 to disable)")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
