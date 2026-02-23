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
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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


def augment(sequence: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Very small augmentation: randomly drop a few time steps (zero them).
    Keep masks consistent."""
    seq = sequence.copy()
    m = mask.copy()
    batch, seq_len, feat = seq.shape
    for i in range(batch):
        # drop up to 5% of timesteps
        k = max(1, int(seq_len * 0.05))
        idx = np.random.choice(seq_len, k, replace=False)
        seq[i, idx, :] = 0
        m[i, idx] = 0
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

    model = StrategyUnsupervisedEncoder(vocabs, embed_out=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        num_batches = int(np.ceil(len(train_idx) / args.batch_size))
        pbar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}", leave=False)
        for seq_b, mask_b, meta_b in batch_generator(train_idx):
            # create two augmentations
            seq1, mask1 = augment(seq_b.numpy(), mask_b.numpy())
            seq2, mask2 = augment(seq_b.numpy(), mask_b.numpy())

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

    # clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    labels = kmeans.fit_predict(all_emb)
    np.savetxt(os.path.join(args.out_dir, "cluster_labels.csv"), labels, fmt="%d")

    # t-SNE plot
    tsne = TSNE(n_components=2, random_state=0)
    proj = tsne.fit_transform(all_emb)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=6)
    plt.colorbar(sc, label="cluster")
    plt.title("Unsupervised clusters (t-SNE)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "clusters_tsne.png"), dpi=150)
    plt.close()

    print(f"Saved labels and plot to {args.out_dir}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="aoe4_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="outputs/unsup")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
