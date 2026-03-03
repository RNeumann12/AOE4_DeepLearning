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
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_unsupervised import StrategyUnsupervisedEncoder

# optional wandb (safe import)
try:
    import wandb
    _wandb_available = True
except Exception:
    wandb = None
    _wandb_available = False


def load_npz(path: str = "aoe4_dataset.npz"):
    """Load dataset from NPZ file."""
    data = np.load(path, allow_pickle=True)
    X_seq = data["X_seq"]
    X_mask = data["X_mask"]
    X_meta = data["X_meta"]
    game_ids = data["game_ids"] if "game_ids" in data else None
    player_ids = data["player_ids"] if "player_ids" in data else None
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

    return X_seq, X_mask, X_meta, vocabs, game_ids, player_ids

# --- clustering helper ---
def cluster_embeddings(embeddings: np.ndarray, method: str, n_clusters: int = 5, random_state: int = 0):
    """
    Cluster embeddings with specified method.
    Returns cluster labels (array of shape [n_samples]).
    """
    method = method.lower()
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state, n_init=1)
        labels = gmm.fit_predict(embeddings)
    elif method == "hdbscan":
        # min_cluster_size can be tuned; here we use 5% of data as default
        min_size = max(2, int(len(embeddings) * 0.05))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10, cluster_selection_epsilon=0)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return labels

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

    X_seq, X_mask, X_meta, vocabs, game_ids, player_ids = load_npz(args.npz)

    time_idx = 4
    vill_idx = 5

    # Only normalize valid tokens
    valid_mask = X_mask.astype(bool)

    # TIME normalization (0–900 → 0–1)
    max_time = 900.0
    X_seq[..., time_idx][valid_mask] = X_seq[..., time_idx][valid_mask] / max_time

    # VILLAGER normalization (scale to 0–1 using max in dataset)
    max_vill = X_seq[..., vill_idx][valid_mask].max()
    if max_vill > 0:
        X_seq[..., vill_idx][valid_mask] = X_seq[..., vill_idx][valid_mask] / max_vill

    print(f"Normalized time by {max_time}, villagers by {max_vill}")

    # simple train/val split by index
    n = X_seq.shape[0]
    idx = np.arange(n)
    np.random.seed(args.seed)
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
    # model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Initialize wandb if requested and available
    use_wandb = getattr(args, 'wandb', False) and _wandb_available
    if getattr(args, 'wandb', False) and not _wandb_available:
        print("wandb requested but not installed; continuing without wandb logging")

    if use_wandb:
        wandb_mode = args.wandb_mode if hasattr(args, 'wandb_mode') else 'online'
        wandb.init(
            project=args.wandb_project,
            # entity=(args.wandb_entity or None),
            # name=(args.wandb_run_name or None),
            config={**vars(args), 'total_params': total_params, 'trainable_params': trainable_params, 'dataset_size': int(n)},
            mode=wandb_mode,
        )
        # store vocab sizes in config
        wandb.config.update({'vocab_sizes': vocabs})
    
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
            embedding, (entity_logits, event_logits, age_logits, time_pred, vill_pred) = model(seq_b, mask_b, meta_b)
# masks
            mask_b = mask_b.to(dtype=torch.float32)  # ensure float for multiplication
            mask_exp = mask_b.unsqueeze(-1)  # shape: [B, seq_len, 1]

            # categorical losses (cross-entropy)
            loss_entity = (F.cross_entropy(
                entity_logits.permute(0, 2, 1),  # [B, classes, seq_len]
                seq_b[:, :, 0].long(),           # target entity indices
                reduction='none'
            ) * mask_b).sum() / mask_b.sum()

            loss_event = (F.cross_entropy(
                event_logits.permute(0, 2, 1),
                seq_b[:, :, 1].long(),
                reduction='none'
            ) * mask_b).sum() / mask_b.sum()

            loss_age = (F.cross_entropy(
                age_logits.permute(0, 2, 1),
                seq_b[:, :, 3].long(),
                reduction='none'
            ) * mask_b).sum() / mask_b.sum()

            mask_exp = mask_b.unsqueeze(-1) 
            # continuous losses (MSE)
            loss_time = ((time_pred - seq_b[:, :, 4:5]) ** 2 * mask_exp).sum() / mask_exp.sum()
            loss_vill = ((vill_pred - seq_b[:, :, 5:6]) ** 2 * mask_exp).sum() / mask_exp.sum()

            # balance losses so continuous features have some effect
            loss = loss_entity + loss_event + loss_age + 10 * (loss_time + loss_vill)

            optimizer.zero_grad()
            loss.backward()
            # gradient norm before clipping (for logging)
            total_norm = 0.0
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            print(loss.item())
            print(loss.item() / args.batch_size)
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

        val_loss = 0.0
        total_val_loss = 0.0
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
                    emb_v, (entity_logits, event_logits, age_logits, time_pred, vill_pred) = model(seq_b, mask_b, meta_b)

                    # masks
                    mask_b = mask_b.to(dtype=torch.float32)  # ensure float for multiplication
                    mask_exp = mask_b.unsqueeze(-1)  # shape: [B, seq_len, 1]

                    # categorical losses (cross-entropy)
                    loss_entity = (F.cross_entropy(
                        entity_logits.permute(0, 2, 1),  # [B, classes, seq_len]
                        seq_b[:, :, 0].long(),           # target entity indices
                        reduction='none'
                    ) * mask_b).sum() / mask_b.sum()

                    loss_event = (F.cross_entropy(
                        event_logits.permute(0, 2, 1),
                        seq_b[:, :, 1].long(),
                        reduction='none'
                    ) * mask_b).sum() / mask_b.sum()

                    loss_age = (F.cross_entropy(
                        age_logits.permute(0, 2, 1),
                        seq_b[:, :, 3].long(),
                        reduction='none'
                    ) * mask_b).sum() / mask_b.sum()

                    # continuous losses (MSE)
                    loss_time = ((time_pred - seq_b[:, :, 4:5]) ** 2 * mask_exp).sum() / mask_exp.sum()
                    loss_vill = ((vill_pred - seq_b[:, :, 5:6]) ** 2 * mask_exp).sum() / mask_exp.sum()

                    # balance losses so continuous features have some effect
                    loss = loss_entity + loss_event + loss_age + 10 * (loss_time + loss_vill)

                
                avg_val_loss = total_val_loss / max(1, steps)
                # Optional dimensionality reduction for stability
                if emb_v.shape[0] > 1000 and emb_v.shape[1] > 10:
                    pca = PCA(n_components=min(50, emb_v.shape[1]))
                    emb_red = pca.fit_transform(emb_v)
                else:
                    emb_red = emb_v

                cluster_method = getattr(args, 'cluster_method', 'kmeans').lower()

                sils = []
                cluster_sizes_runs = []

                for r in range(args.silhouette_repeats):
                    labr  = cluster_embeddings(all_emb, args.cluster_method, args.n_clusters)
                    n_clusters_found = len(set(labr)) - (1 if -1 in labr else 0)
                    # compute silhouette only if more than 1 cluster
                    if len(set(labr)) > 1:
                        silr = silhouette_score(emb_red, labr)
                    else:
                        silr = float('nan')
                    sils.append(silr)

                    # optional: record cluster sizes
                    counts = np.array([(labr == k).sum() for k in range(args.n_clusters)])
                    cluster_sizes_runs.append(counts.tolist())

                # compute cluster size statistics
                cluster_sizes_arr = np.array(cluster_sizes_runs)
                cluster_sizes_mean = cluster_sizes_arr.mean(axis=0).tolist()
                cluster_sizes_std = cluster_sizes_arr.std(axis=0).tolist()

                # Compute silhouette statistics (using last run)
                sil_mean = float(np.nanmean(sils))
                sil_std = float(np.nanstd(sils))
                if sil_std == 0.0:
                    sil_std = None
                silhouette_history.append(sil_mean)
                np.save(os.path.join(args.out_dir, "silhouette_history.npy"), np.array(silhouette_history))

                # Embedding variance
                emb_var = float(np.mean(np.std(emb_v, axis=0)))

                # Additional cluster quality metrics (use last KMeans run labels)
                # Calinski-Harabasz Index (higher is better)
                ch_score = float(calinski_harabasz_score(emb_red, labr)) if len(set(labr)) > 1 else float('nan')
                # Davies-Bouldin Index (lower is better)
                db_score = float(davies_bouldin_score(emb_red, labr)) if len(set(labr)) > 1 else float('nan')
                
                # Per-cluster reconstruction variance: capture per-cluster homogeneity
                per_cluster_var = {}
                try:
                    for c in range(args.n_clusters):
                        c_mask = (labr == c)
                        if c_mask.sum() > 0:
                            # use embedding variance within cluster
                            c_emb = emb_v[c_mask]
                            c_var = float(np.mean(np.var(c_emb, axis=0)))
                            per_cluster_var[f'cluster_{c}_embedding_var'] = c_var
                except Exception:
                    pass
                
                # Pairwise cosine similarity: mean and variance (checks for collapse)
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    cos_sim = cosine_similarity(emb_v)
                    # mask out diagonal
                    cos_sim_no_diag = cos_sim[~np.eye(cos_sim.shape[0], dtype=bool)]
                    mean_cos_sim = float(np.mean(cos_sim_no_diag))
                    var_cos_sim = float(np.var(cos_sim_no_diag))
                except Exception:
                    mean_cos_sim = float('nan')
                    var_cos_sim = float('nan')
                
                # Cluster stability: Adjusted Rand Index between two KMeans runs
                try:
                    km1 = KMeans(n_clusters=args.n_clusters, random_state=10, n_init=10)
                    km2 = KMeans(n_clusters=args.n_clusters, random_state=11, n_init=10)
                    lab1 = km1.fit_predict(emb_red)
                    lab2 = km2.fit_predict(emb_red)
                    ari = float(adjusted_rand_score(lab1, lab2))
                except Exception:
                    ari = float('nan')

                # Record epoch metrics into JSONL (repeat after t-SNE)
                em = {
                    'epoch': epoch + 1,
                    'silhouette_mean': sil_mean,
                    'silhouette_std': sil_std,
                    'cluster_sizes_mean': cluster_sizes_mean,
                    'cluster_sizes_std': cluster_sizes_std,
                    'embedding_var_mean': emb_var,
                    'sample_n': int(sample_n),
                    'avg_loss': float(avg_loss),
                    'last_batch_grad_norm': float(total_norm) if 'total_norm' in locals() else None
                }
                with open(os.path.join(args.out_dir, 'epoch_metrics.jsonl'), 'a') as fh:
                    fh.write(json.dumps(em) + "\n")

                print(f"  silhouette (sample {sample_n}): mean={sil_mean:.4f}" + (f" std={sil_std:.4f}" if sil_std is not None else " std=None"))

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

                # log to wandb at epoch end if enabled
                if use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/avg_loss': float(avg_loss),
                        'eval/avg_loss': float(avg_val_loss),
                        'silhouette_mean': sil_mean,
                        'embedding_var_mean': emb_var,
                        'dataset_size': int(n),
                        'last_batch_grad_norm': float(total_norm) if 'total_norm' in locals() else None,
                    }
                    # cluster size fields
                    for k, v in enumerate(cluster_sizes_mean):
                        log_dict[f'cluster_{k}_size_mean'] = float(v)
                    # if sil_std is not None:
                    #     log_dict['silhouette_std'] = sil_std
                    # # add advanced cluster quality metrics
                    # if 'ch_score' in locals():
                    #     log_dict['calinski_harabasz'] = ch_score
                    # if 'db_score' in locals():
                    #     log_dict['davies_bouldin'] = db_score
                    # if 'per_cluster_var' in locals():
                    #     log_dict.update(per_cluster_var)
                    # if 'mean_cos_sim' in locals():
                    #     log_dict['mean_cosine_similarity'] = mean_cos_sim
                    # if 'var_cos_sim' in locals():
                    #     log_dict['var_cosine_similarity'] = var_cos_sim
                    # if 'ari' in locals():
                    #     log_dict['cluster_stability_ari'] = ari
                    # attach cluster image if present
                    # if args.save_epoch_tsne and sample_n > 0:
                    #     img_path = os.path.join(args.out_dir, f"clusters_epoch_{epoch+1}.png")
                    #     if os.path.exists(img_path):
                    #         log_dict['epoch_cluster'] = wandb.Image(img_path)
                    try:
                        wandb.log(log_dict, step=epoch+1)
                    except Exception as e:
                        print(f"wandb epoch logging failed: {e}")

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

    # Concatenate all embeddings into single array
    all_emb = np.vstack(all_emb)
    
    # Save embeddings
    np.save(os.path.join(args.out_dir, 'embeddings.npy'), all_emb)
    print(f"Saved embeddings: {all_emb.shape}")

    # --- Final clustering based on CLUSTER_METHOD ---
    CLUSTER_METHOD = getattr(args, 'cluster_method', 'kmeans').lower()  # either "kmeans" or "gmm"
    print(f"\nFinal clustering using method: {CLUSTER_METHOD.upper()} with K={args.n_clusters}...")

    sils_final = []
    labels_runs = []

    for r in range(args.cluster_repeats):
        labels_final = cluster_embeddings(all_emb, args.cluster_method, args.n_clusters)

        unique, counts = np.unique(labels_final, return_counts=True)
        print(dict(zip(unique, counts)))

        n_clusters_found = len(set(labels_final)) - (1 if -1 in labels_final else 0)
        print(f"  Run {r+1}/{args.cluster_repeats} - found {n_clusters_found} clusters")
        labels_runs.append(labels_final)
        
        # compute silhouette if more than 1 cluster present
        if len(set(labels_final)) > 1:
            sil = silhouette_score(all_emb, labels_final)
        else:
            sil = float('nan')
        sils_final.append(sil)

    sil_final_mean = float(np.nanmean(sils_final))
    sil_final_std = float(np.nanstd(sils_final))

    # Save cluster assignments with game_id and player_id
    cluster_labels = labels_runs[-1]
    if game_ids is not None and player_ids is not None:
        df_clusters = pd.DataFrame({
            'game_id': game_ids,
            'player_id': player_ids,
            'cluster': cluster_labels
        })
        df_clusters.to_csv(os.path.join(args.out_dir, f"cluster_labels_{CLUSTER_METHOD}.csv"), index=False)
    else:
        np.savetxt(os.path.join(args.out_dir, f"cluster_labels_{CLUSTER_METHOD}.csv"), cluster_labels, fmt="%d")

    with open(os.path.join(args.out_dir, f'{CLUSTER_METHOD}_silhouette_summary.json'), 'w') as fh:
        json.dump({'mean': sil_final_mean, 'std': sil_final_std, 'repeats': args.cluster_repeats}, fh)

    print(f"Final silhouette score ({CLUSTER_METHOD}): {sil_final_mean:.4f} ± {sil_final_std:.4f}")

    # t-SNE visualization
    print("Generating t-SNE visualization...")
    try:

        # Compute 2D projection
        proj = TSNE(n_components=2, random_state=0).fit_transform(all_emb)

        # Simple scatter plot
        plt.scatter(proj[:, 0], proj[:, 1], c=labels_runs[-1], cmap="tab10", s=6)  # s=10 keeps dots small
        plt.colorbar(label="cluster")
        plt.title(f"t-SNE clusters ({CLUSTER_METHOD})")
        plt.show()
        plt.savefig(os.path.join(args.out_dir, f"clusters_{CLUSTER_METHOD}.png"), dpi=150)
        plt.close()
        print("Saved t-SNE plot")
    except Exception as e:
        print(f"t-SNE plot failed: {e}")

    # Log final artifacts to wandb
    if _wandb_available and getattr(args, 'wandb', False):
        try:
            final_tsne = os.path.join(args.out_dir, f"clusters_{CLUSTER_METHOD}.png")
            to_log = {}
            if os.path.exists(final_tsne):
                to_log['final'] = wandb.Image(final_tsne)
            # also save embeddings and label file
            emb_path = os.path.join(args.out_dir, 'embeddings.npy')
            labels_path = os.path.join(args.out_dir, f'cluster_labels_{CLUSTER_METHOD}.csv')
            if os.path.exists(emb_path):
                wandb.save(emb_path)
            if os.path.exists(labels_path):
                wandb.save(labels_path)
            if len(to_log) > 0:
                wandb.log(to_log)
        except Exception as e:
            print(f"wandb final artifact logging failed: {e}")


def cli():
    parser = argparse.ArgumentParser(description="Train unsupervised autoencoder for AoE4 clustering")
    parser.add_argument("--npz", type=str, default="aoe4_dataset.npz", help="Path to NPZ dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--out_dir", type=str, default="outputs/unsup", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"], help="Device")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on silhouette")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--silhouette_repeats", type=int, default=5, help="KMeans repeats for validation silhouette")
    parser.add_argument("--cluster_repeats", type=int, default=1, help="KMeans repeats for final clustering")
    parser.add_argument("--seq_dropout", type=float, default=0.1, help="Sequence dropout rate")
    parser.add_argument("--proj_dropout", type=float, default=0.1, help="Projection dropout rate")
    parser.add_argument("--silhouette_sample", type=int, default=1000, help="Validation samples for silhouette (0 to disable)")
    parser.add_argument("--save_epoch_tsne", action="store_true", default=True, help="Save a t-SNE plot and cluster labels for each epoch")
    parser.add_argument("--csv_path", type=str, default="input_event_based.csv", help="Path to original training CSV for cluster analysis (optional)")
    # wandb logging args
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging if available")
    parser.add_argument("--wandb_project", type=str, default="StrategyDiscovery-Unsupervised-V2", help="WandB project name")
    # parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/user name")
    # parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="WandB mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cluster_method", type=str, default="hdbscan", help="Clustering method (kmeans or gmm)")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
