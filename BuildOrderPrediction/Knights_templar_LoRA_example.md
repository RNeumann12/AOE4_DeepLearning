# LoRA Patch Adaptation — Knights Templar Example

**Matchup:** Knights Templar vs French · Dry Arabia
**Patch:** 15.3.8338
**Adapter:** `lora_patch_15.3.8338.pth` (val_loss = 0.4541)

---

## What changed

French is known for powerful early cavalry. Against the base model, Knights Templar respond late
with a long dark-age Spearman rush before eventually transitioning into Hospitaller Knights and
cavalry in Feudal. After patch 15.3.8338, the LoRA adapter learns a fundamentally different response:
**rush in Dark Age, age up much faster, then add Archers to counter the French answer against Spearman (Archer).**

| | Base Model | + LoRA Adapter |
|---|---|---|
| First Barracks | Step 3 | Step 2 (one step earlier) |
| Age-up | Step 34–35 (very late) | Step 22–23 (12 steps earlier) |
| Feudal military buildings | Stable | Archery Range |
| Feudal composition | Spearman 2 + Hospitaller Knight + Horseman | Spearman 2 + Archer 2 |
| Strategy | Hospitaller Knight (heal Spearman) | Archer 2 (counter French Archer) |

The most notable shift is the age-up timing: the LoRA adapter commits to Feudal **12 steps earlier**,
then opens an Archery Range instead of trying to heal Spearman

---

## Adapter stats

```
Trainable parameters:  1,047,160  (0.14% of 733M total)
LoRA tensors loaded:   54  (27 adapters × 2 matrices)
Rank / Alpha:          8 / 16.0
Target modules:        ffn_only (FFN layers + output head)
```

---

## Commands

```bash
# With patch 15.3.8338 LoRA adapter
python BuildOrderPrediction/MoE_WithDecoder_infer.py \
    --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth \
    --lora_checkpoint BuildOrderPrediction/lora_patch_15.3.8338.pth \
    --player_civ knights_templar --enemy_civ French \
    --map "Dry Arabia" --build_steps 50

# Base model — no adapter
python BuildOrderPrediction/MoE_WithDecoder_infer.py \
    --checkpoint BuildOrderPrediction/MoE_WithDecoder_best_model.pth \
    --player_civ knights_templar --enemy_civ French \
    --map "Dry Arabia" --build_steps 50
```

---

## Build order comparison

```
Step  + LoRA Adapter (patch 15.3.8338)    Base Model (no adapter)
────  ─────────────────────────────────   ─────────────────────────────────
 1    Villager                            Villager
 2    Barracks                  ←rush     Villager
 3    Villager                            Barracks
 4    Spearman 1                          House
 5    House                               Spearman 1
 6    Villager                            Villager
 7    Spearman 1                          Spearman 1
 8    Villager                            Villager
 9    Spearman 1                          Spearman 1
10    Villager                            Spearman 1
11    Spearman 1                          Villager
12    Villager                            Spearman 1
13    Gold Mining Camp                    Villager
14    Villager                            Spearman 1
15    Spearman 1                          Villager
16    Villager                            House
17    House                               Villager
18    Villager                            Spearman 1
19    Villager                            Spearman 1
20    Spearman 1                          Villager
21    Spearman 1                          Spearman 1
22    Civ Icon (Principality)  ←age-up    Spearman 1
23    Age Display Persistent 2            Spearman 1
24    Spearman 1                          Spearman 1
25    Villager                            Villager
26    Spearman 1                          Spearman 1
27    Spearman 2               ←Upgrade   Villager
28    Spearman 2                          Spearman 1
29    Archery Range            ←Archers   Spearman 1
30    Spearman 2                          Spearman 1
31    Villager                            Villager
32    Archer 2                            House
33    Spearman 2                          Spearman 1
34    Archer 2                            Civ Icon (Knight Hospitalier)  ←age-up
35    Villager                            Age Display Persistent 2
36    Archer 2                            Villager
37    Spearman 2                          Spearman 2
38    Archer 2                            Safepassage
39    Archer 2                            Hospitaller Knight Age 2
40    Spearman 2                          Villager
41    Archer 2                            Spearman 2
42    Villager                            Stable                         ←cavalry
43    Spearman 2                          Villager
44    House                               Spearman 2
45    Spearman 2                          Horseman 2                     ←cavalry
46    Villager                            Hospitaller Knight Age 2
47    Spearman 2                          Safepassage
48    Archer 2                            Spearman 2
49    Spearman 2                          Horseman 2
50    Villager                            Villager
```

---

## Confidence score (LoRA adapter)

```
Overall Confidence:    35.29%
Geometric Mean Prob:   0.3529
Perplexity:            2.83   (lower = more confident)
Mean Log Probability:  -1.0417

Min probability:       6.37%
Max probability:       91.86%
Median probability:    36.93%

Interpretation: Moderate confidence — Build order follows common patterns
```
