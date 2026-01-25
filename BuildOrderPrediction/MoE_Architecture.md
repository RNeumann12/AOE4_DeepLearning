# Enhanced Sequence Predictor with Mixture of Experts (MoE)

---

## High-Level System Architecture

```mermaid
flowchart TB
    subgraph Data["Data Pipeline"]
        CSV["CSV Input"] --> Vocabs["build_vocabularies"]
        CSV --> Dataset["SequenceDataset"]
        Vocabs --> Dataset
        Dataset --> Filter["Filters Applied<br/>(DESTROY events, Sheep, timestamp=0)"]
        Filter --> GameSplit["Game-Level Split"]
        GameSplit --> TrainLoader["Train DataLoader"]
        GameSplit --> ValLoader["Val DataLoader"]
    end

    subgraph Model["SequencePredictor (Enhanced)"]
        Embeddings["Embeddings<br/>(Entity, Civ, Map)"]
        NGram["N-Gram Extractor"]
        RoPE["Rotary Position Embeddings"]
        CrossAttn["Gated Cross-Attention"]
        Transformer["Improved Transformer Blocks"]
        MoE["Mixture of Experts"]
        Heads["Output Head"]
    end

    subgraph Training["Training Loop"]
        Trainer["SequencePredictorTrainer"]
        Augmenter["SequenceAugmenter"]
        Losses["Multi-Task Losses"]
        Optim["AdamW + AMP + Scheduler"]
    end

    TrainLoader --> Augmenter
    Augmenter --> Trainer
    Model --> Trainer
    Trainer --> Losses
    Losses --> Optim

    ValLoader --> Evaluate["evaluate()"]
    Model --> Evaluate
    Evaluate --> Metrics["Top-K Accuracy Metrics"]
```

---

## Model Architecture: SequencePredictor

```mermaid
flowchart TB
    subgraph Inputs["Inputs"]
        EntitySeq["entity_sequence<br/>(B, L)"]
        PC["player_civ<br/>(B,)"]
        EC["enemy_civ<br/>(B,)"]
        MapID["map_id<br/>(B,)"]
    end

    subgraph Embeddings["Embedding Layers"]
        EntityEmb["Entity Embedding<br/>(vocab_size → d_model)"]
        PCivEmb["Player Civ Embedding"]
        ECivEmb["Enemy Civ Embedding"]
        MapEmb["Map Embedding"]
    end

    subgraph ConditionFusion["Condition Fusion"]
        CondConcat["Concatenate<br/>(3 × d_model)"]
        CondProj["Projection MLP<br/>(3d → 2d → d)"]
        GatedCrossAttn["Gated Cross-Attention"]
    end

    subgraph FeatureExtraction["Feature Extraction"]
        NGram["N-Gram Feature Extractor<br/>(1-gram to 4-gram)"]
        RoPE["Rotary Position Embeddings"]
    end

    subgraph TransformerStack["Transformer Stack (8 layers)"]
        Block0["Block 0: Local Attention"]
        MoE0["MoE Layer (Expert Selection)"]
        Block1["Block 1: Global Attention"]
        Block2["Block 2: Local Attention"]
        MoE1["MoE Layer"]
        BlockN["... Alternating Pattern"]
    end

    subgraph OutputHead["Output Head"]
        OutLN["Layer Norm"]
        DeepHead["Deep FFN<br/>(d → 2d → d)"]
        GatedSkip["Gated Skip Connection"]
        Classifier["Entity Classifier<br/>(d → vocab_size)"]
    end

    subgraph AuxOutputs["Auxiliary Outputs"]
        ContrastiveProj["Contrastive Projection<br/>(d → d/2)"]
    end

    EntitySeq --> EntityEmb
    PC --> PCivEmb
    EC --> ECivEmb
    MapID --> MapEmb

    PCivEmb --> CondConcat
    ECivEmb --> CondConcat
    MapEmb --> CondConcat
    CondConcat --> CondProj

    EntityEmb --> NGram
    NGram --> RoPE
    RoPE --> GatedCrossAttn
    CondProj --> GatedCrossAttn

    GatedCrossAttn --> Block0
    Block0 --> MoE0
    MoE0 --> Block1
    Block1 --> Block2
    Block2 --> MoE1
    MoE1 --> BlockN

    BlockN --> OutLN
    OutLN --> DeepHead
    OutLN --> GatedSkip
    DeepHead --> GatedSkip
    GatedSkip --> Classifier
    OutLN --> ContrastiveProj
```

---

## Key Components

### 1. Rotary Position Embeddings (RoPE)

```mermaid
flowchart LR
    Input["Input (B, L, d)"] --> Split["Split into pairs"]
    Split --> Rotate["Apply Rotation Matrix"]
    Rotate --> Output["Output (B, L, d)"]
    
    style Rotate fill:#4a9eff
```

RoPE encodes relative positions through rotation matrices, enabling:
- Better generalization to longer sequences
- Position-aware attention without additive embeddings
- Efficient computation via precomputed cos/sin caches

### 2. Gated Cross-Attention

```mermaid
flowchart LR
    X["Entity Repr (B, L, d)"] --> CrossAttn["Multi-Head<br/>Cross-Attention"]
    Cond["Condition (B, 1, d)"] --> CrossAttn
    CrossAttn --> AttnOut["Attention Output"]
    X --> Concat["Concat"]
    AttnOut --> Concat
    Concat --> Gate["Sigmoid Gate"]
    Gate --> GatedResidual["Gated Residual"]
    X --> GatedResidual
    GatedResidual --> Output["Output (B, L, d)"]
    
    style Gate fill:#ff9f4a
```

The gate learns to control how much condition information (player civ, enemy civ, map) flows into the entity representations.

### 3. N-Gram Feature Extractor

```mermaid
flowchart TB
    Input["Input (B, L, d)"] --> Conv1["1-gram Conv1D"]
    Input --> Conv2["2-gram Conv1D"]
    Input --> Conv3["3-gram Conv1D"]
    Input --> Conv4["4-gram Conv1D"]
    
    Conv1 --> Concat["Concatenate"]
    Conv2 --> Concat
    Conv3 --> Concat
    Conv4 --> Concat
    
    Concat --> Project["Linear Projection"]
    Project --> LayerNorm["Layer Norm + Residual"]
    LayerNorm --> Output["Output (B, L, d)"]
```

Captures local patterns in build orders:
- 1-gram: Individual entity importance
- 2-gram: Adjacent pairs (e.g., "Villager → Farm")
- 3-gram: Common sequences (e.g., "Villager, Villager, House")
- 4-gram: Extended patterns (e.g., "Barracks → Spearman × 3")

### 4. Mixture of Experts (MoE)

```mermaid
flowchart TB
    Input["Input (B, L, d)"] --> Router["Router Network<br/>(Linear → num_experts)"]
    Router --> TopK["Top-K Selection<br/>(K=2 by default)"]
    
    TopK --> Expert1["Expert 1<br/>(Economy?)"]
    TopK --> Expert2["Expert 2<br/>(Military?)"]
    TopK --> Expert3["Expert 3<br/>(Tech?)"]
    TopK --> Expert4["Expert 4<br/>(Timing?)"]
    TopK --> ExpertN["Expert N<br/>(...)"]
    
    Expert1 --> WeightedSum["Weighted Sum<br/>(by router probs)"]
    Expert2 --> WeightedSum
    Expert3 --> WeightedSum
    Expert4 --> WeightedSum
    ExpertN --> WeightedSum
    
    WeightedSum --> Residual["Residual + LayerNorm"]
    Residual --> Output["Output (B, L, d)"]
    
    Router --> AuxLoss["Load Balancing<br/>Auxiliary Loss"]
    
    style TopK fill:#9f4aff
    style AuxLoss fill:#ff4a4a
```

| Expert | Potential Specialization |
|--------|-------------------------|
| Expert 1 | Economy decisions (Villagers, resource buildings) |
| Expert 2 | Military units (Barracks, Archery Range, units) |
| Expert 3 | Technology/Upgrades |
| Expert 4 | Timing-based decisions |
| Expert 5 | Civilization-specific strategies |
| Expert 6 | Counter-play / Reactions |

The **load balancing loss** encourages even expert usage:
$$L_{aux} = \alpha \sum_{e=1}^{N} (p_e - \frac{1}{N})^2$$

### 5. Improved Transformer Block

```mermaid
flowchart TB
    Input["Input"] --> PreNorm1["Pre-LayerNorm"]
    PreNorm1 --> SelfAttn["Multi-Head<br/>Self-Attention"]
    SelfAttn --> Dropout1["Dropout"]
    Input --> Gate1["Gate α"]
    Dropout1 --> Gate1
    Gate1 --> Residual1["Residual"]
    
    Residual1 --> LocalAttn["Local Attention<br/>(window=8)<br/>Every 2nd Layer"]
    
    LocalAttn --> PreNorm2["Pre-LayerNorm"]
    PreNorm2 --> FFN["FFN<br/>(GELU)"]
    LocalAttn --> Gate2["Gate β"]
    FFN --> Gate2
    Gate2 --> Output["Output"]
    
    style Gate1 fill:#4aff9f
    style Gate2 fill:#4aff9f
    style LocalAttn fill:#ffff4a
```

Key improvements:
- **Pre-norm architecture** for stable deep training
- **Gated residual connections** with learnable gates (α, β)
- **Alternating local/global attention** for multi-scale patterns

---

## Loss Functions

```mermaid
pie title Loss Composition
    "Focal Loss (Entity)" : 1.0
    "MoE Auxiliary" : 0.01
    "Contrastive" : 0.05
```

### Loss Details

| Loss | Weight | Formula | Description |
|------|--------|---------|-------------|
| **Focal Loss** | 1.0 | $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$ | Addresses class imbalance, γ=2.0 |
| **MoE Aux** | 0.01 | $\sum_e (p_e - 1/N)^2$ | Load balancing across experts |
| **Contrastive** | 0.05 | InfoNCE | Same-civ sequences → similar embeddings |

### Focal Loss Visualization

```
                      ┌─────────────────────────────────────┐
                      │  Focal Loss: Focus on Hard Examples │
                      ├─────────────────────────────────────┤
 Loss Weight          │                                     │
       ▲              │  ●  Cross-Entropy                   │
       │              │  ○  Focal Loss (γ=2)                │
   1.0 ├──────●───────│─────────────────────────────────────│
       │       \      │                                     │
   0.8 ├────────●─────│                                     │
       │         \    │                                     │
   0.6 ├──────────●───│                                     │
       │           \  │                                     │
   0.4 ├────────────●─│                                     │
       │             \│      Well-classified examples       │
   0.2 ├──────────────●────●───●───●───────────────────────│
       │               \   │   │   │   (down-weighted)      │
   0.0 ├────────────────●──●───●───●───●───●───●──────────►│
       └──────────────────────────────────────────────────►
       0.0           0.25    0.5   0.75    1.0
                           p_t (Prediction Confidence)
```

---

## Data Pipeline

### SequenceDataset

```mermaid
flowchart TB
    subgraph Preprocessing["Preprocessing"]
        Raw["Raw CSV"] --> FilterEvents["Filter DESTROY events"]
        FilterEvents --> FilterEntities["Filter Sheep<br/>(captured, not built)"]
        FilterEntities --> FilterTime["Filter timestamp=0<br/>(starting units)"]
        FilterTime --> FilterWins["Filter wins_only<br/>(optional)"]
    end
    
    subgraph Grouping["Sequence Creation"]
        FilterWins --> GroupBy["Group by<br/>(game_id, profile_id)"]
        GroupBy --> SortTime["Sort by time"]
        SortTime --> Chunk["Chunk to max_seq_len"]
        Chunk --> AddBOS["Prepend ⟨BOS⟩ token"]
        AddBOS --> Pad["Pad to fixed length"]
    end
    
    subgraph Splitting["Game-Level Split"]
        Pad --> UniqueGames["Get unique game_ids"]
        UniqueGames --> StratifiedSplit["Stratified Split<br/>(85% train / 15% val)"]
        StratifiedSplit --> TrainSubset["Train Subset"]
        StratifiedSplit --> ValSubset["Val Subset"]
    end
```

### Sequence Augmentation

| Augmentation | Probability | Description |
|-------------|-------------|-------------|
| **Adjacent Swap** | 5% | Swap two adjacent entities |
| **Drop** | 2% | Remove a random entity |
| **Repeat** | 2% | Duplicate an entity |

---

## Training Features

### Teacher Forcing Schedule

```mermaid
flowchart LR
    A["TF = 100%<br/>Always Ground Truth"] 
    
    style A fill:#2d5a27,color:#fff
```

Default: **100% teacher forcing** for stable training with this enhanced architecture.

### Learning Rate Schedule

```mermaid
flowchart LR
    A["Warmup<br/>(8 epochs)"] --> B["Cosine Decay"] --> C["Min LR = 1%"]
    
    style A fill:#ff9f4a
    style B fill:#4a9eff
    style C fill:#9f4aff
```

$$LR(t) = \begin{cases} 
\frac{t}{t_{warmup}} \cdot LR_{max} & t < t_{warmup} \\
LR_{max} \cdot \max(0.01, 0.5(1 + \cos(\pi \cdot \frac{t - t_{warmup}}{t_{total} - t_{warmup}}))) & t \geq t_{warmup}
\end{cases}$$

### Differential Learning Rates

| Component | Learning Rate |
|-----------|---------------|
| Embeddings | `lr × 0.3` |
| Attention & FFN | `lr × 1.0` |

---

## Civ-Entity Masking

```mermaid
flowchart LR
    subgraph Training["During Training/Inference"]
        Logits["Entity Logits<br/>(B, vocab_size)"]
        CivMask["Civ-Entity Mask<br/>(num_civs, vocab_size)"]
        PlayerCiv["Player Civ ID"]
        
        PlayerCiv --> Lookup["Lookup Mask Row"]
        CivMask --> Lookup
        Lookup --> BatchMask["Batch Mask<br/>(B, vocab_size)"]
        
        Logits --> Apply["Apply Mask<br/>(-∞ for invalid)"]
        BatchMask --> Apply
        Apply --> MaskedLogits["Masked Logits"]
    end
```

This ensures the model can only predict entities that the player's civilization can actually build.

---

## Entity Class Weights

Log-dampened inverse frequency weighting:

$$w_e = \frac{\log(N_{total} / N_e)}{\log(N_{total} / N_{median})}$$

```
┌────────────────────────────────────────────────────────────┐
│                Entity Class Weight Distribution             │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ Villager        ███████████████████████████████ 0.35       │
│ Farm            ████████████████████████ 0.55              │
│ House           ███████████████████ 0.65                   │
│ Lumber Camp     ████████████████ 0.78                      │
│ ...                                                        │
│ Trebuchet       ████████████████████████████████████ 2.1   │
│ Wonder          ████████████████████████████████████ 2.5   │
│ Unique Unit     ████████████████████████████████████ 2.8   │
│                                                             │
│              Low Weight ◄──────────► High Weight           │
│              (Common)                (Rare)                 │
└────────────────────────────────────────────────────────────┘
```

Weights are capped at `[0.2, 5.0]` and normalized to mean = 1.0.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Exact match of predicted entity |
| **Top-3 Accuracy** | Correct entity in top 3 predictions |
| **Top-5 Accuracy** | Correct entity in top 5 predictions |
| **Top-10 Accuracy** | Correct entity in top 10 predictions |
| **Mean Per-Class Acc** | Average accuracy across all entity types |
| **Loss** | Focal loss on validation set |

---

## Code Structure

| File | Key Components |
|------|----------------|
| [MoE_train.py](MoE_train.py) | Main training script |

### Classes

| Class | Purpose |
|-------|---------|
| `SequencePredictor` | Enhanced Transformer model with MoE, RoPE, N-gram |
| `SequencePredictorTrainer` | Training loop with contrastive + focal loss |
| `MixtureOfExperts` | Expert routing and weighted aggregation |
| `RotaryPositionalEmbedding` | RoPE implementation |
| `GatedCrossAttention` | Condition fusion with learnable gate |
| `NGramFeatureExtractor` | Multi-scale conv1D patterns |
| `ImprovedTransformerBlock` | Pre-norm + gated residuals |
| `LocalAttentionBlock` | Windowed attention (size=8) |
| `FocalLoss` | Class-imbalance aware loss |
| `ContrastiveLoss` | InfoNCE for representation learning |
| `SequenceAugmenter` | Data augmentation strategies |
| `SequenceDataset` | Game-level data loading |

### Utility Functions

| Function | Purpose |
|----------|---------|
| `build_vocabularies()` | Create entity/civ/map vocabs |
| `compute_entity_class_weights()` | Log-dampened frequency weights |
| `build_civ_entity_mapping()` | Map civs to valid entities |
| `create_civ_entity_mask()` | Boolean mask tensor |
| `create_data_loaders()` | Game-level train/val split |
| `log_example_predictions()` | WandB prediction logging |

---

## Default Hyperparameters

```mermaid
graph LR
    %% Left side - Architecture & Training
    a1((d_model<br/>448)) --- A[🏗️ Arch]
    a2((nhead<br/>8)) --- A
    a3((layers<br/>8)) --- A
    a4((ffn<br/>1792)) --- A
    
    t1((epochs<br/>100)) --- T[🎯 Train]
    t2((batch<br/>160)) --- T
    t3((lr<br/>2e-4)) --- T
    t4((accum<br/>3)) --- T
    
    A --- Model((🎮<br/>MoE))
    T --- Model
    
    %% Right side - Optimizer & Features
    Model --- O[⚡ Optim]
    Model --- F[🎛️ Features]
    
    O --- o1((AdamW))
    O --- o2((wd<br/>0.05))
    O --- o3((AMP ✓))
    O --- o4((warmup<br/>8))
    
    F --- f1((experts<br/>6))
    F --- f2((RoPE ✓))
    F --- f3((ngram ✓))
    F --- f4((top_k<br/>2))
    
    %% Top - Regularization
    r1((smooth<br/>0.1)) --- R[🛡️ Reg]
    r2((focal<br/>γ=2)) --- R
    r3((contrast<br/>0.05)) --- R
    r4((augment<br/>0.2)) --- R
    R --- Model

    style Model fill:#c0392b,color:#fff,stroke:#fff,stroke-width:3px
    style A fill:#2980b9,color:#fff
    style T fill:#27ae60,color:#fff
    style O fill:#f39c12,color:#fff
    style R fill:#8e44ad,color:#fff
    style F fill:#16a085,color:#fff

    style a1 fill:#3498db,color:#000
    style a2 fill:#3498db,color:#000
    style a3 fill:#3498db,color:#000
    style a4 fill:#3498db,color:#000

    style t1 fill:#2ecc71,color:#000
    style t2 fill:#2ecc71,color:#000
    style t3 fill:#2ecc71,color:#000
    style t4 fill:#2ecc71,color:#000

    style o1 fill:#f1c40f,color:#000
    style o2 fill:#f1c40f,color:#000
    style o3 fill:#f1c40f,color:#000
    style o4 fill:#f1c40f,color:#000

    style r1 fill:#9b59b6,color:#fff
    style r2 fill:#9b59b6,color:#fff
    style r3 fill:#9b59b6,color:#fff
    style r4 fill:#9b59b6,color:#fff

    style f1 fill:#1abc9c,color:#000
    style f2 fill:#1abc9c,color:#000
    style f3 fill:#1abc9c,color:#000
    style f4 fill:#1abc9c,color:#000
```

### Parameter Reference Tables

| Architecture | Value |
|--------------|-------|
| d_model | 448 |
| nhead | 8 |
| num_layers | 8 |
| dim_feedforward | 1792 (4× d_model) |
| max_seq_len | 128 |
| dropout | 0.15 |
| num_experts | 6 |

| Training | Value |
|----------|-------|
| batch_size | 160 |
| grad_accum_steps | 3 (effective: 480) |
| learning_rate | 2e-4 |
| epochs | 100 |
| warmup_epochs | 8 |
| val_split | 0.15 |
| weight_decay | 0.05 |

| Loss & Regularization | Value |
|----------------------|-------|
| label_smoothing | 0.1 |
| focal_loss_gamma | 2.0 |
| contrastive_weight | 0.05 |
| augment_prob | 0.2 |
| moe_aux_loss_coef | 0.01 |

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--use_moe` | True | Enable Mixture of Experts |
| `--use_ngram` | True | Enable N-gram feature extraction |
| `--use_rope` | True | Use Rotary Position Embeddings |
| `--use_contrastive` | True | Contrastive learning auxiliary loss |
| `--use_augmentation` | True | Sequence augmentation during training |
| `--wins_only` | False | Train only on winning games |

---

## Inference: Autoregressive Generation

```mermaid
flowchart LR
    subgraph Init["Initialization"]
        BOS["⟨BOS⟩"] --> Seq["[⟨BOS⟩]"]
    end
    
    subgraph Loop["Generation Loop"]
        Seq --> Model["SequencePredictor"]
        Model --> Logits["Next Entity Logits"]
        Logits --> Temp["Temperature Scaling"]
        Temp --> TopK["Top-K Filtering"]
        TopK --> TopP["Nucleus (Top-P)"]
        TopP --> Sample["Sample"]
        Sample --> Append["Append to Sequence"]
        Append --> Seq
    end
    
    subgraph Output["Output"]
        Append --> Generated["Generated Build Order"]
    end
```

Sampling parameters:
- `temperature`: Controls randomness (1.0 = neutral)
- `top_k`: Keep only top-k most likely tokens
- `top_p`: Nucleus sampling threshold

---

## Usage

```bash
# Basic training
python BuildOrderPrediction/MoE_train.py

# Train on winning games only
python BuildOrderPrediction/MoE_train.py --wins_only

# Custom configuration
python BuildOrderPrediction/MoE_train.py \
    --epochs 150 \
    --batch_size 128 \
    --d_model 512 \
    --num_layers 10 \
    --num_experts 8 \
    --lr 1e-4

# Disable WandB logging
python BuildOrderPrediction/MoE_train.py --no_wandb
```

---

## Model Checkpointing

### Saved Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'entity_vocab': entity_vocab,
    'civ_vocab': civ_vocab,
    'map_vocab': map_vocab,
    'civ_entity_mapping': {k: list(v) for k, v in civ_entity_mapping.items()},
    'args': vars(args),
    'val_metrics': final_metrics,
    'best_val_loss': best_val_loss,
    'architecture': 'EnhancedSequencePredictor'
}
```

Files saved:
- `best_model.pth` - Best validation loss model
- `final_model.pth` - Final model with full metadata
