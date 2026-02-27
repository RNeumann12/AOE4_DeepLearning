"""Train `StrategyUnsupervisedEncoder` with a simple contrastive loss,
then cluster embeddings and save assignments + t-SNE plot.

Notes:
- Uses data from `aoe4_dataset.npz` (same loading as supervised trainer)
- Default objective: NT-Xent (normalized temperature-scaled cross entropy)
- No augmentation is implemented beyond simple time-window noise; this
  is intentionally minimal—replace or extend augment() for better results.
"""
from __future__ import annotations
import os
import argparse
from typing import Tuple

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


from torch.utils.data import DataLoader, Dataset
from dataset_event_based import AoE4Dataset


def load_npz(path: str = "aoe4_dataset.npz"):
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


def augment(sequence: np.ndarray, mask: np.ndarray, dropout_prob: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Augment by dropping random timesteps and token dropout.
    - timestep dropout: zero a small fraction of timesteps (contiguous or random)
    - token/entity dropout: with `dropout_prob` randomly zero entity/event tokens
    Returns augmented (seq, mask).
    """
    seq = sequence.copy()
    m = mask.copy()
    batch, seq_len, feat = seq.shape

    # timestep dropout: drop up to 5% timesteps per sample
    for i in range(batch):
        k = max(1, int(seq_len * 0.05))
        idx = np.random.choice(seq_len, k, replace=False)
        seq[i, idx, :] = 0
        m[i, idx] = 0

    # token/entity dropout: randomly zero out entity/event columns per timestep
    # entity at index 0, event at index 1
    if dropout_prob > 0:
        mask_tokens = np.random.rand(batch, seq_len) < dropout_prob
        # zero entity and event tokens on masked positions
        seq[mask_tokens, 0] = 0
        seq[mask_tokens, 1] = 0

    return seq, m


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5):
    # z_i, z_j: (B, D) normalized
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature
    # mask out self-similarity
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    # remove diagonal
    mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)).float()
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1)

    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -torch.log(pos / denom)
    return loss.mean()


def collate_from_arrays(X_seq, X_mask, X_meta, indices):
    seq = torch.from_numpy(X_seq[indices]).float()
    mask = torch.from_numpy(X_mask[indices]).float()
    meta = torch.from_numpy(X_meta[indices]).long()
    return seq, mask, meta


def train(args):
    # device selection: allow forcing 'cuda' or 'cpu', or use 'auto'
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    X_seq, X_mask, X_meta, vocabs = load_npz(args.npz)

    # simple split by index
    n = X_seq.shape[0]
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(n * (1 - args.val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Dataloaders (we'll use simple batching via indices)
    def batch_generator(indices):
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            seq, mask, meta = collate_from_arrays(X_seq, X_mask, X_meta, batch_idx)
            yield seq, mask, meta

    model = StrategyUnsupervisedEncoder(vocabs, embed_out=args.embed_dim, seq_dropout=args.seq_dropout, proj_dropout=args.proj_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            # create two augmentations or use minimal jitter when augmentation disabled
            if args.augment:
                seq1, mask1 = augment(seq_b.numpy(), mask_b.numpy(), dropout_prob=args.dropout)
                seq2, mask2 = augment(seq_b.numpy(), mask_b.numpy(), dropout_prob=args.dropout)
            else:
                # minimal jitter: keep sequences identical except tiny noise on continuous cols (time, villagers)
                seq_np = seq_b.numpy()
                seq1 = seq_np.copy()
                seq2 = seq_np.copy()
                # add tiny gaussian noise to time (col 4) and villagers (col 5)
                noise = np.random.normal(scale=1e-3, size=seq2[:, :, 4:6].shape)
                seq2[:, :, 4:6] = seq2[:, :, 4:6] + noise
                mask1 = mask_b.numpy()
                mask2 = mask_b.numpy()
                seq1 = torch.from_numpy(seq1).to(device).float()
                seq2 = torch.from_numpy(seq2).to(device).float()
                mask1 = torch.from_numpy(mask1).to(device)
                mask2 = torch.from_numpy(mask2).to(device)
            meta = meta_b.to(device)

            z1 = model(seq1, mask1, meta)
            z2 = model(seq2, mask2, meta)

            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            # update progress
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        avg_loss = total_loss / max(1, steps)
        loss_history.append(avg_loss)
        # persist loss history each epoch
        os.makedirs(args.out_dir, exist_ok=True)
        np.save(os.path.join(args.out_dir, "loss_history.npy"), np.array(loss_history))
        print(f"Epoch {epoch+1}/{args.epochs} - avg_loss: {avg_loss:.4f}")

        # Silhouette monitoring on a validation subset
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
                    emb_v = model(seq_v, mask_v, meta_v).cpu().numpy()

                # optionally reduce dimensionality for stability
                if emb_v.shape[0] > 1000 and emb_v.shape[1] > 10:
                    pca = PCA(n_components=min(50, emb_v.shape[1]))
                    emb_red = pca.fit_transform(emb_v)
                else:
                    emb_red = emb_v

                km = KMeans(n_clusters=args.n_clusters, random_state=0)
                lab = km.fit_predict(emb_red)
                # repeat KMeans to get mean/std silhouette and cluster sizes
                sils = []
                cluster_sizes_runs = []
                for r in range(args.silhouette_repeats):
                    kmr = KMeans(n_clusters=args.n_clusters, random_state=r)
                    labr = kmr.fit_predict(emb_red)
                    if len(set(labr)) > 1:
                        silr = silhouette_score(emb_red, labr)
                    else:
                        silr = float('nan')
                    sils.append(silr)
                    # cluster sizes
                    counts = np.array([ (labr == k).sum() for k in range(args.n_clusters) ])
                    cluster_sizes_runs.append(counts.tolist())

                sil_mean = float(np.nanmean(sils))
                sil_std = float(np.nanstd(sils))
                silhouette_history.append(sil_mean)
                np.save(os.path.join(args.out_dir, "silhouette_history.npy"), np.array(silhouette_history))

                # embedding variance per-dim and mean
                emb_var = float(np.mean(np.std(emb_v, axis=0)))

                # cluster sizes summary (mean and std across repeats)
                cluster_sizes_arr = np.array(cluster_sizes_runs)
                cluster_sizes_mean = cluster_sizes_arr.mean(axis=0).tolist()
                cluster_sizes_std = cluster_sizes_arr.std(axis=0).tolist()

                # record epoch metrics
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
                # save metrics
                with open(os.path.join(args.out_dir, 'epoch_metrics.jsonl'), 'a') as fh:
                    fh.write(json.dumps(em) + "\n")

                print(f"  silhouette (sample {sample_n}): mean={sil_mean:.4f} std={sil_std:.4f}")

                # early stopping based on silhouette
                if not np.isnan(sil_mean) and sil_mean > best_sil:
                    best_sil = sil_mean
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # save best model
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

    # after training, compute embeddings for all games
    model.eval()
    with torch.no_grad():
        all_emb = []
        batch_meta = []
        for i in range(0, n, args.batch_size):
            seq = torch.from_numpy(X_seq[i : i + args.batch_size]).float().to(device)
            mask = torch.from_numpy(X_mask[i : i + args.batch_size]).to(device)
            meta = torch.from_numpy(X_meta[i : i + args.batch_size]).long().to(device)
            emb = model(seq, mask, meta)
            all_emb.append(emb.cpu().numpy())
            batch_meta.append(meta.cpu().numpy())

    all_emb = np.vstack(all_emb)
    os.makedirs(args.out_dir, exist_ok=True)

    # save embeddings
    np.save(os.path.join(args.out_dir, 'embeddings.npy'), all_emb)

    # clustering with KMeans (baseline) repeated
    sils_final = []
    labels_runs = []
    for r in range(args.cluster_repeats):
        kmf = KMeans(n_clusters=args.n_clusters, random_state=r)
        labf = kmf.fit_predict(all_emb)
        labels_runs.append(labf)
        if len(set(labf)) > 1:
            sils_final.append(silhouette_score(all_emb, labf))
        else:
            sils_final.append(float('nan'))

    sil_final_mean = float(np.nanmean(sils_final))
    sil_final_std = float(np.nanstd(sils_final))
    # save last labels and summary
    np.savetxt(os.path.join(args.out_dir, "cluster_labels_kmeans.csv"), labels_runs[-1], fmt="%d")
    with open(os.path.join(args.out_dir, 'kmeans_silhouette_summary.json'), 'w') as fh:
        json.dump({'mean': sil_final_mean, 'std': sil_final_std, 'repeats': args.cluster_repeats}, fh)

    # t-SNE plot of KMeans clusters
    try:
        tsne = TSNE(n_components=2, random_state=0)
        proj = tsne.fit_transform(all_emb)
        plt.figure(figsize=(8, 6))
        # use the last KMeans run labels for plotting
        sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels_runs[-1], cmap="tab10", s=6)
        plt.colorbar(sc, label="cluster")
        plt.title("KMeans clusters (t-SNE)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "clusters_kmeans_tsne.png"), dpi=150)
        plt.close()
    except Exception:
        import traceback
        print("t-SNE plot failed; skipping t-SNE visualization")
        traceback.print_exc()

    # Try UMAP + HDBSCAN for density-based clustering (optional)
    # Disabled due to missing packages; uncomment if umap-learn and hdbscan are installed
    # try:
    #     import umap
    #     import hdbscan
    #
    #     print("Running UMAP + HDBSCAN...")
    #     reducer = umap.UMAP(n_components=min(10, all_emb.shape[1]), random_state=0)
    #     emb_umap = reducer.fit_transform(all_emb)
    #
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=args.hdb_min_cluster_size, min_samples=args.hdb_min_samples)
    #     hdb_labels = clusterer.fit_predict(emb_umap)
    #     np.savetxt(os.path.join(args.out_dir, "cluster_labels_hdbscan.csv"), hdb_labels, fmt="%d")
    #
    #     # 2D visualization (UMAP to 2D)
    #     reducer2 = umap.UMAP(n_components=2, random_state=1)
    #     emb2d = reducer2.fit_transform(all_emb)
    #     plt.figure(figsize=(8, 6))
    #     sc = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=hdb_labels, cmap="tab10", s=6)
    #     plt.colorbar(sc, label="hdbscan_cluster")
    #     plt.title("HDBSCAN clusters (UMAP 2D)")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(args.out_dir, "clusters_hdbscan_umap.png"), dpi=150)
    #     plt.close()
    #     print(f"Saved HDBSCAN labels and UMAP plot to {args.out_dir}")
    # except Exception as e:
    #     print(f"UMAP+HDBSCAN skipped (missing packages or error): {e}")


    print(f"Saved KMeans labels and plots to {args.out_dir}")

    # plot loss + silhouette history
    try:
        plt.figure(figsize=(8, 4))
        epochs = np.arange(1, len(loss_history) + 1)
        plt.plot(epochs, loss_history, label='loss')
        if len(silhouette_history) > 0:
            plt.plot(epochs[: len(silhouette_history)], silhouette_history, label='silhouette')
        plt.xlabel('epoch')
        plt.legend()
        plt.title('Loss and Silhouette over epochs')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'loss_silhouette.png'))
        plt.close()
    except Exception:
        pass


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="aoe4_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="outputs/unsup")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable data augmentations during contrastive training")
    parser.add_argument("--dropout", type=float, default=0.3, help="Token/entity dropout probability used when augmenting")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs) on silhouette")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--silhouette_repeats", type=int, default=5, help="Number of KMeans repeats for silhouette mean/std during validation")
    parser.add_argument("--cluster_repeats", type=int, default=10, help="Number of KMeans repeats for final clustering stability")
    parser.add_argument("--seq_dropout", type=float, default=0.1, help="Sequence input dropout in the encoder")
    parser.add_argument("--proj_dropout", type=float, default=0.3, help="Projection head dropout in the encoder")
    parser.add_argument("--silhouette_sample", type=int, default=1000, help="Number of validation samples to compute silhouette per epoch (0 to disable)")
    parser.add_argument("--hdb_min_cluster_size", type=int, default=20)
    parser.add_argument("--hdb_min_samples", type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
