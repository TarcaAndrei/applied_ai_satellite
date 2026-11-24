# Results

All EuroSAT numbers use the fixed stratified 70/15/15 split (`splits/eurosat_ms_seed42.csv`, seed 42): fit on
train (18,900), choose settings on val (4,050), score test (4,050) once. Run outputs are in `runs/<name>/`, which
is gitignored, so the numbers that matter are copied here.

## 2026-11-24 · B1: XGBoost on spectral features (EuroSAT)

**Main result: XGBoost reaches 94.6% test accuracy (macro-F1 0.943) from 82 per-patch summary features, without
seeing the patch layout.** That is a high floor: the CNN and DINOv2 models have to beat it to justify themselves on
EuroSAT.

### Setup

- **Features** (`sat/features.py`, 82 in total). Uses only the 12 bands that Cluj's L2A images also have (no B10),
  selected by band name.
  - 60 band features: each band's mean, std and 10th/50th/90th percentiles of reflectance.
  - 20 index features: NDVI, NDWI, MNDWI, NDBI and NDRE, each with mean, std, p10 and p90.
  - 2 texture features: average edge strength of B08 and of RGB brightness.
  - None of them change when a patch is flipped or rotated, so no augmentation is needed.
- **Model** (`scripts/train_xgb.py`). XGBoost with learning rate 0.05, subsample 0.8 and colsample 0.8. Trees
  are added until val log-loss stops improving for 100 rounds. Depth (4/6/8) and `min_child_weight` (1/5) were
  chosen on val accuracy. The best was depth 4, `min_child_weight` 1, 907 trees. Training takes about 15 s on CPU.
- **Comparisons** on the same split and features: the majority class (B0) and a standardised logistic regression.

### Test results

| Model | Features | Test acc | Macro-F1 |
|---|---|---|---|
| B0 majority class | none | 0.111 | 0.020 |
| Logistic regression | all 82 | 0.922 | 0.918 |
| **XGBoost** | **all 82** | **0.946** | **0.943** |
| Logistic regression | 20 index features | 0.817 | 0.807 |
| XGBoost | 20 index features | 0.894 | 0.888 |

Val accuracy for XGBoost with all features is 0.946, the same as test, so choosing settings on val did not overfit.

Test F1 for each class, XGBoost with all features:

| Class | F1 | | Class | F1 |
|---|---|---|---|---|
| Forest | 0.990 | | PermanentCrop | 0.933 |
| SeaLake | 0.990 | | AnnualCrop | 0.932 |
| HerbaceousVegetation | 0.974 | | Pasture | 0.923 |
| Residential | 0.969 | | **Highway** | **0.822** |
| River | 0.953 | | | |
| Industrial | 0.946 | | | |

The most frequent test errors: Highway → Industrial (18), AnnualCrop → PermanentCrop (17), Highway → AnnualCrop
(17), Industrial → Highway (14), Highway → PermanentCrop (12), Residential → Highway (11).

The features with the highest gain: NDVI_p10, MNDWI_mean, B02_std, B8A_mean, NDWI_std, B11_mean.

### Conclusions

1. **Spectrum alone explains most of EuroSAT.** Without any spatial layout, a tree model gets 94.6% and a linear
   model gets 92.2%. Most classes are separable by colour and vegetation/water indices. Part of this is the random
   split: neighbouring patches from the same scene end up in both train and test (see PLAN.md), which makes it easy.
2. **The remaining errors are about shape, not colour.** Highway is the weakest class (F1 0.82). It and the
   AnnualCrop/PermanentCrop pair are where summary statistics run out, because a road or a row of orchard trees
   is a geometric pattern. This is where the CNN and DINOv2 should gain. If they don't improve these classes, they
   add little over B1.
3. **Raw reflectance is worth about 5 points on EuroSAT** (0.946 all features vs 0.894 indices only; 0.922 vs
   0.817 for logistic regression). The model relies on absolute brightness levels. These are exactly what will
   shift between EuroSAT (L1C, top of atmosphere) and Cluj (L2A, surface reflectance, plus the +1000 offset after
   2022). **Prediction to test in Phase 4:** the all-features model will lose more on Cluj than the indices-only
   one, and the gap between them may shrink or reverse. Both models are saved, so this costs only running the
   features on the Cluj patches.
4. **Trees beat the linear model by about 2 points with all features and about 8 points with indices only.** The
   relationships between features and classes are non-linear, especially when only ratios are available.

### Caveats

- Single split, single seed; no confidence intervals yet. With 4,050 test patches, the standard error of accuracy
  is about ±0.4 points, so differences under about 1 point are not meaningful.
- The texture features are only 2 crude edge-strength measures. A richer texture set (e.g. GLCM) would likely
  narrow the Highway gap and should be tried before claiming the deep models win on shape.

### Reproduce

```
python -m scripts.train_xgb --tune                                 # runs/xgb_b1_all
python -m scripts.train_xgb --features index --max-depth 4         # runs/xgb_b1_index
```

## 2026-11-24 · M2: ResNet-18 from random initialisation (EuroSAT)

**Main result: ResNet-18 trained from random weights on the 12 L2A bands reaches 98.7% test accuracy (macro-F1
0.987), 4.1 points above XGBoost. Errors fall from 217 to 52 of 4,050. The gain is largest on Highway (F1 0.82 →
0.98), the class B1 predicted would need spatial layout.**

### Setup

- **Model** (`sat/models.py`). torchvision ResNet-18, `weights=None`, 11.2M parameters. The stem is adapted to
  64 px patches: conv1 is 3×3, stride 1, with 12 input bands, and the max-pool after it is removed, so the last
  stage sees 8×8 instead of 2×2.
- **Input.** The 12 bands that Cluj's L2A images also have (no B10), each standardised with the train-split
  mean/std (`splits/eurosat_ms_band_stats_seed42.json`). Augmentation: random flips and 90° rotations.
- **Training** (`scripts/train_cnn.py`). AdamW, weight decay 5e-4, batch 256, 30 epochs, 2 warm-up epochs then
  cosine decay, bf16 mixed precision, plain cross-entropy (no label smoothing, so the probabilities stay usable
  for the change-detection thresholds). The epoch with the best val macro-F1 is kept.
- **Learning rate** chosen on val: 1e-3 (val macro-F1 0.987, best epoch 27) over 3e-4 (0.985, epoch 23). Within
  noise of each other. About 4.5 minutes in total for both runs on the A100.

### Test results

| Model | Input | Test acc | Macro-F1 |
|---|---|---|---|
| B0 majority class | none | 0.111 | 0.020 |
| B1 XGBoost | 82 spectral features | 0.946 | 0.943 |
| **M2 ResNet-18, random init** | **12 bands, 64×64** | **0.987** | **0.987** |

Val accuracy is 0.988, test 0.987. Train accuracy at the end is 99.6%, so there is only a small train/val gap.

Test F1 for each class:

| Class | ResNet-18 | XGBoost | Gain |
|---|---|---|---|
| **Highway** | **0.981** | **0.822** | **+0.159** |
| AnnualCrop | 0.982 | 0.932 | +0.050 |
| Pasture | 0.980 | 0.923 | +0.057 |
| PermanentCrop | 0.981 | 0.933 | +0.048 |
| Industrial | 0.989 | 0.946 | +0.043 |
| River | 0.984 | 0.953 | +0.031 |
| Residential | 0.996 | 0.969 | +0.027 |
| HerbaceousVegetation | 0.983 | 0.974 | +0.009 |
| Forest | 0.996 | 0.990 | +0.006 |
| SeaLake | 0.994 | 0.990 | +0.004 |

The most frequent test errors (52 in total): HerbaceousVegetation → AnnualCrop (5), River → SeaLake (5),
AnnualCrop → PermanentCrop (4), Pasture → HerbaceousVegetation (4), Highway → Industrial (3).

### Conclusions

1. **Spatial layout is worth about 4 points on EuroSAT, and it goes where B1 predicted.** The classes that gain
   most are the ones defined by shape: Highway (+16 F1 points), then the crop/pasture classes (+5) and Industrial
   (+4). Classes that colour already separates (Forest, SeaLake, HerbaceousVegetation) gain under 1 point.
2. **The remaining errors are between classes that genuinely look alike**: herbaceous vegetation vs crops, river
   vs lake, annual vs permanent crops. No single class stands out any more.
3. **This number is optimistic for the same reason as B1's, probably more so.** With the random split,
   neighbouring patches of the same scene are in train and test, and a CNN can match their textures directly.
4. **Prediction to test in Phase 4:** like XGBoost with all features, the CNN sees absolute reflectance
   (standardised with EuroSAT L1C statistics), so the L1C → L2A shift on Cluj should hurt it. Whether it loses
   more or less than B1 is one of the gap experiments.

### Caveats

- Single split, single seed. At 98.7% on 4,050 patches the standard error is about ±0.2 points, so the gap to
  XGBoost (4.1 points) is real, but the 3e-4 vs 1e-3 difference is not.
- The optional 13-band variant (with B10) has not been run yet: `python -m scripts.train_cnn --bands all`.

### Reproduce

```
python -m scripts.train_cnn --lr 3e-4 1e-3                         # runs/resnet18_random_l2a12
```
