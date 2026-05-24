# LightGBM — Complete Interview Guide

## 5-Pillar Answer Framework

| Pillar | Answer |
|--------|--------|
| **WHAT** | LightGBM (Light Gradient Boosting Machine) is Microsoft's gradient boosting framework optimised for speed and memory efficiency via **leaf-wise tree growth**, **histogram-based split finding**, **Gradient-based One-Side Sampling (GOSS)**, and **Exclusive Feature Bundling (EFB)**. |
| **WHY** | It trains up to 20× faster than XGBoost and uses significantly less memory, while maintaining comparable or superior accuracy. These gains enable training on very large datasets and rapid iteration during model development. |
| **WHEN** | Use LightGBM when: dataset is large (>100K samples), training speed is critical, memory is limited, or features are sparse/high-dimensional. It is the dominant algorithm in Kaggle tabular competitions as of 2020+. |
| **HOW** | Converts continuous features to discrete histograms (256 bins by default). Grows trees leaf-wise (best-first), targeting the leaf with maximum delta loss. Applies GOSS to skip low-gradient samples and EFB to bundle mutually exclusive features, dramatically reducing computation. |
| **PITFALLS** | Leaf-wise growth can overfit on small datasets (use `num_leaves` and `min_data_in_leaf` carefully); more sensitive to hyperparameters than XGBoost; categorical feature handling can be unexpected; default parameters tuned for large data — need adjustment for small datasets. |

---

## Leaf-Wise vs Level-Wise Tree Growth

This is LightGBM's most distinctive architectural choice.

### Level-Wise Growth (XGBoost, sklearn GBM)

Grows the tree level by level — all nodes at depth d are split before any node at depth d+1:

```
Depth 0:          [Root]
                  /     \
Depth 1:       [A]       [B]          ← both A and B split next
              / \         / \
Depth 2:    [C] [D]    [E] [F]        ← all four split next
```

- **Advantage:** Balanced trees, less risk of overfitting.
- **Disadvantage:** Many splits on unimportant nodes waste computation.

### Leaf-Wise Growth (LightGBM)

At each step, find the single leaf with the **maximum delta loss** (gain) and split only that leaf:

```
Start:        [Root]
              /     \
Round 1:   [A]       [B]              ← B has highest gain, split B
              /     \     \
Round 2:   [A]    [E]  [F]            ← E has highest gain, split E
           /  \  /  \   \
Round 3: [G][H][I][J] [F]             ← keep splitting best leaf
```

- **Advantage:** More accuracy per split; focuses computation where it matters.
- **Disadvantage:** Can create unbalanced trees that overfit on small data.

```
Visual comparison for same n_leaves:

Level-wise (depth=3):     Leaf-wise (max_leaves=8):
        ●                         ●
       / \                       / \
      ●   ●                     ●   ●
     /|   |\                       / \
    ● ● ● ● ●                     ●   ●
    (8 leaves, balanced)              / \
                                     ●   ●
                                     (8 leaves, skewed)
```

**Key parameter:** Use `num_leaves` instead of `max_depth` as the primary complexity control in LightGBM. Rule of thumb: `num_leaves < 2^max_depth`.

---

## GOSS: Gradient-based One-Side Sampling

### The Problem

In standard gradient boosting, all n training samples are used to find splits at every round. This is expensive when n is large.

### The Insight

Samples with large gradients are the ones the model currently fits poorly — they carry the most information for improving the model. Samples with small gradients are already well-fitted and contribute less.

### GOSS Algorithm

```
At each boosting round:
  1. Sort samples by |gradient|
  2. Keep top-a fraction (large gradient samples) — always include these
  3. From remaining (1-a) fraction, randomly sample b fraction
  4. Amplify retained small-gradient samples by factor (1-a)/b
     to compensate for downsampling

Retained samples: a·n + b·(1-a)·n  << n
Amplification: w_small = (1-a)/b   (corrects for sampling bias)
```

**Example:**
```
n = 1,000,000 samples
a = 0.2  → keep top 200,000 (large gradients)
b = 0.1  → random sample 10% of remaining 800,000 = 80,000

Total used: 200,000 + 80,000 = 280,000  (only 28% of data!)
Amplification factor: (1-0.2)/0.1 = 8.0
```

**Effect:** Significant speedup (3–5×) with minimal accuracy loss because the excluded samples are already well-fitted.

### GOSS Theoretical Bound

Approximation error from GOSS vs using all data is bounded:

```
|Ṽ - V̄| ≤ C · (1-a)/(n·b)
```

where V̄ is the true split gain and Ṽ is the GOSS estimate. The error decreases with more retained samples.

---

## EFB: Exclusive Feature Bundling

### The Problem

High-dimensional datasets (especially with one-hot encoded categoricals) are typically **sparse** — many features are zero simultaneously. Building histograms for all features separately is wasteful.

### The Insight

Features that rarely take non-zero values simultaneously are **mutually exclusive** and can be bundled into a single feature without loss of information.

### EFB Algorithm

**Step 1: Build conflict graph**
- Features are nodes; an edge connects two features if they have conflicting (simultaneously non-zero) values.
- Find near-exclusive bundles by solving an approximate graph colouring problem.

**Step 2: Bundle features**
For exclusive features A (range [0, 10]) and B (range [0, 20]):
```
Bundled feature C:
  If A is non-zero: C = A           (range 0–10)
  If B is non-zero: C = B + 10      (range 10–30)
  If both zero:     C = 0
```

By shifting the ranges, the original values are recoverable from the bundle.

**Step 3: Build histogram for bundle C instead of A and B separately.**

**Effect:**
- Reduces effective feature count from p to much fewer bundles.
- Speedup scales with sparsity: 10× speedup reported on high-dimensional sparse data.

---

## Histogram Binning Speed Advantage

### The Core Optimisation

Instead of finding splits on exact feature values, LightGBM discretises features into K bins (default K=256):

```
Continuous feature X: [1.2, 3.5, 0.4, 7.8, 2.1, ...]

Binned: compute thresholds [0.5, 1.0, 1.5, ..., 8.0] (255 thresholds)
        Map each value to bin: [2, 6, 0, 15, 3, ...]

Histogram: Count(bin_k) = number of samples with X in bin k
           Sum_grad(bin_k) = Σ gᵢ for samples in bin k
           Sum_hess(bin_k) = Σ hᵢ for samples in bin k
```

### Finding the Best Split

```
For each threshold between bins k and k+1:
  G_L = Σ Sum_grad(0..k),   H_L = Σ Sum_hess(0..k)
  G_R = G_total - G_L,      H_R = H_total - H_L
  Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ

Cost: O(K) per feature per node  (K=256 bins, much faster than O(n) exact)
```

### Histogram Subtraction Trick

For a node with known parent histogram, the sibling histogram can be computed by subtraction:

```
hist(left child) = computed from scratch
hist(right child) = hist(parent) - hist(left child)  ← free!
```

This halves the histogram computation cost at each node.

---

## LightGBM vs XGBoost Full Comparison

| Dimension | LightGBM | XGBoost |
|-----------|----------|---------|
| **Tree growth** | Leaf-wise (best-first) | Level-wise |
| **Split finding** | Histogram (bins) | Exact or approximate |
| **Speed** | 🟢 ~20× faster (large data) | 🟡 Slower but more tunable |
| **Memory** | 🟢 Much lower (histogram bins) | 🟡 Higher |
| **Small data** | 🟡 Can overfit, needs careful tuning | 🟢 More robust |
| **Categorical features** | 🟢 Native optimal handling | 🟡 Requires encoding |
| **GPU support** | ✅ Yes | ✅ Yes |
| **Distributed training** | ✅ Yes | ✅ Yes |
| **Missing values** | ✅ Yes (native) | ✅ Yes (native) |
| **Sparse features** | 🟢 EFB compresses sparse | 🟡 Manual handling better |
| **Primary depth control** | `num_leaves` | `max_depth` |
| **Regularisation params** | `min_data_in_leaf`, `lambda_l1/l2`, `min_gain_to_split` | `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight` |
| **Typical accuracy** | 🟢 Slightly better (large data) | 🟡 Similar |
| **Community/ecosystem** | 🟢 Very active | 🟢 Very active, older |
| **Default hyperparams** | Tuned for large data | More universal |

---

## Complete Parameter Guide

### Core Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_estimators` | 100 | 100–5000 | Number of boosting rounds |
| `learning_rate` | 0.1 | 0.01–0.3 | Shrinkage rate |
| `num_leaves` | 31 | 20–300 | Max leaves per tree (primary complexity control) |
| `max_depth` | -1 | -1 (unlimited), 3–12 | Hard depth limit (-1 = no limit) |
| `min_data_in_leaf` | 20 | 10–1000 | Min samples in leaf (critical for overfitting) |

### Regularisation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lambda_l1` | 0.0 | 0–10 | L1 regularisation |
| `lambda_l2` | 0.0 | 0–10 | L2 regularisation |
| `min_gain_to_split` | 0.0 | 0–5 | Min gain to make a split |
| `bagging_fraction` | 1.0 | 0.5–1.0 | Row subsampling (requires `bagging_freq > 0`) |
| `bagging_freq` | 0 | 0, 1–10 | Frequency of bagging (0 = disabled) |
| `feature_fraction` | 1.0 | 0.3–1.0 | Column subsampling per tree |

### GOSS & EFB Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `boosting_type` | 'gbdt' | 'gbdt', 'dart', 'goss', 'rf' |
| `top_rate` | 0.2 | GOSS: fraction of large-gradient samples kept |
| `other_rate` | 0.1 | GOSS: fraction of small-gradient samples kept |
| `max_bin` | 255 | Number of histogram bins |

---

## Complete Python Implementation

```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ── 1. Data preparation ───────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ── 2. Native LightGBM API ────────────────────────────────────────────────────
dtrain = lgb.Dataset(X_tr,   label=y_tr,   feature_name=feature_names)
dval   = lgb.Dataset(X_val,  label=y_val,  reference=dtrain)

params = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'learning_rate':     0.05,
    'num_leaves':        31,
    'max_depth':         -1,
    'min_data_in_leaf':  20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'lambda_l1':         0.1,
    'lambda_l2':         1.0,
    'min_gain_to_split': 0.0,
    'verbose':           -1,
    'seed':              42,
}

callbacks = [
    early_stopping(stopping_rounds=50, verbose=False),
    log_evaluation(period=100),
]

bst = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'val'],
    callbacks=callbacks,
)

y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
print("═" * 50)
print("LIGHTGBM NATIVE API")
print("═" * 50)
print(f"Best iteration : {bst.best_iteration}")
print(f"Test ROC-AUC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, (y_pred_proba >= 0.5).astype(int),
                             target_names=data.target_names))

# ── 3. sklearn API ────────────────────────────────────────────────────────────
lgbm_clf = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,   # = min_data_in_leaf in sklearn API
    subsample=0.8,          # = bagging_fraction
    subsample_freq=5,       # = bagging_freq
    colsample_bytree=0.8,   # = feature_fraction
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_clf.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(50, verbose=False)]
)
y_proba_sk = lgbm_clf.predict_proba(X_test)[:, 1]
print(f"\nsklearn API — Test AUC: {roc_auc_score(y_test, y_proba_sk):.4f}")
print(f"Best iteration       : {lgbm_clf.best_iteration_}")

# ── 4. Feature importance ─────────────────────────────────────────────────────
for imp_type in ['split', 'gain']:
    importances = bst.feature_importance(importance_type=imp_type)
    sorted_idx  = np.argsort(importances)[::-1][:5]
    print(f"\nTop-5 Features (importance_type='{imp_type}'):")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {feature_names[idx]:<35s} {importances[idx]:.1f}")

# ── 5. Cross-validation with lgb.cv ──────────────────────────────────────────
dall = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
cv_results = lgb.cv(
    params,
    dall,
    num_boost_round=1000,
    nfold=5,
    stratified=True,
    shuffle=True,
    callbacks=[early_stopping(30, verbose=False)],
    seed=42,
)

best_n = len(cv_results['valid auc-mean'])
best_auc = max(cv_results['valid auc-mean'])
print(f"\n5-fold CV — Best n_estimators: {best_n}, AUC: {best_auc:.4f}")

# ── 6. GOSS boosting ─────────────────────────────────────────────────────────
goss_params = {**params, 'boosting_type': 'goss', 'top_rate': 0.2, 'other_rate': 0.1}
goss_bst = lgb.train(
    goss_params, dtrain, num_boost_round=500,
    valid_sets=[dval], callbacks=[early_stopping(50, verbose=False),
                                   log_evaluation(period=-1)]
)
y_goss = goss_bst.predict(X_test)
print(f"\nGOSS AUC: {roc_auc_score(y_test, y_goss):.4f}")

# ── 7. Categorical feature handling ──────────────────────────────────────────
# LightGBM can handle categoricals natively — much better than one-hot encoding
# for high-cardinality features:
#
#   df['cat_feature'] = df['cat_feature'].astype('category')
#   lgb.Dataset(X, categorical_feature=['cat_feature'])
#
# LightGBM uses optimal many-vs-many split finding for categoricals:
# O(k log k) vs O(2^k) naive.
print("\n[Categorical] LightGBM handles cat features natively via lgb.Dataset(...,"
      " categorical_feature=[...])")
```

---

## Interview Q&A

**Q1: Why is leaf-wise growth faster and more accurate than level-wise?**

> Leaf-wise growth focuses computation on the leaf with the highest potential gain, avoiding wasted splits on leaves that wouldn't reduce loss significantly. This achieves lower loss with fewer splits. However, it can create highly unbalanced trees that overfit small datasets. Mitigation: use `num_leaves` (not `max_depth`) as the primary complexity control, and set appropriate `min_data_in_leaf`.

**Q2: Explain GOSS and why it preserves accuracy despite sampling.**

> GOSS keeps all high-gradient samples (large errors) and randomly samples a small fraction of low-gradient samples (already well-fitted). The low-gradient samples are amplified by `(1-a)/b` to correct for the sampling bias. The high-gradient samples are informationally critical for finding good splits; the low-gradient samples contribute little. GOSS achieves ~3–5× speedup with accuracy loss bounded by the sampling ratio.

**Q3: What is EFB and when does it help most?**

> EFB bundles mutually exclusive features (those that never have non-zero values simultaneously) into single features, reducing the effective feature count without information loss. It helps most on sparse high-dimensional data — particularly datasets with many one-hot encoded categorical variables, NLP feature matrices, or any data where most features are zero. EFB can achieve 10×+ speedup on sparse datasets.

**Q4: How do `num_leaves` and `max_depth` interact in LightGBM?**

> `num_leaves` is the primary complexity control. `max_depth` is a hard limit on depth but has limited effect in leaf-wise growth. A common mistake is setting `max_depth=6` thinking it controls complexity like in XGBoost. In LightGBM, set `num_leaves` first (31 is default; try 20–150), and only use `max_depth` to prevent extremely imbalanced paths. Rule of thumb: `num_leaves < 2^(max_depth-1)` to avoid overfitting.

**Q5: How does LightGBM handle categorical features differently from XGBoost?**

> LightGBM has native categorical feature support. For a categorical feature with k categories, it uses an efficient optimal-split algorithm with O(k log k) complexity that finds the best partition of categories into two groups (like gradient-weighted Fisher's method). XGBoost treats categoricals as numeric, requiring encoding. LightGBM's approach is much more powerful for high-cardinality categoricals and avoids the curse of one-hot encoding dimensionality.

**Q6: LightGBM or XGBoost — how do you choose in practice?**

> **LightGBM** for: large datasets (>100K rows), high-dimensional features, time-sensitive training, native categorical features, sparse data with EFB. **XGBoost** for: small-to-medium datasets where overfitting risk is high, when level-wise growth is preferred for stability, or when the team is already familiar with its hyperparameters. **Start with LightGBM** for most tabular ML problems today; it is faster and often more accurate. Use XGBoost as a comparison or in ensembles.

---

## Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| Overfitting on small datasets | Leaf-wise growth, default params tuned for large data | Reduce `num_leaves`, increase `min_data_in_leaf`, add regularisation |
| Ignoring `min_data_in_leaf` | Default=20 may be too low | Increase to 50–200 for small data |
| Using `max_depth` instead of `num_leaves` | XGBoost habit | Primary control is `num_leaves` in LightGBM |
| Forgetting `bagging_freq` | `bagging_fraction` alone doesn't enable bagging | Set `bagging_freq=5` when using `bagging_fraction < 1` |
| Not encoding categoricals properly | LightGBM needs explicit `categorical_feature` parameter | Cast to `category` dtype or pass `categorical_feature` list |
| Very high `num_leaves` without regularisation | Rapid overfitting | Always pair high `num_leaves` with `lambda_l2` and `min_data_in_leaf` |

---

## Quick Reference Cheat Sheet

```
LIGHTGBM — QUICK REFERENCE
═══════════════════════════════════════════════════════════

KEY INNOVATIONS
  1. Leaf-wise growth     → best-first splits, faster convergence
  2. Histogram binning    → O(K) vs O(n) split finding
  3. GOSS                 → sample high-gradient + few low-gradient
  4. EFB                  → bundle mutually exclusive sparse features

KEY HYPERPARAMETERS
  num_leaves        → primary complexity (31 default, 20–150)
  min_data_in_leaf  → critical for overfitting on small data (20+)
  learning_rate     → 0.05–0.1 with early stopping
  feature_fraction  → 0.7–0.9 (colsample_bytree equivalent)
  bagging_fraction  → 0.7–0.9 (+ bagging_freq=5)
  lambda_l1/l2      → regularisation
  max_bin           → 255 (fewer bins = faster, slightly less accurate)

EARLY STOPPING
  lgb.train(..., callbacks=[early_stopping(50)])

TYPICAL STARTING PARAMS (large data)
  num_leaves=63, learning_rate=0.05, min_data_in_leaf=50,
  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
  lambda_l2=1.0, n_estimators=2000

VS XGBOOST
  LightGBM param        ↔  XGBoost param
  num_leaves            ↔  controlled via max_depth
  min_data_in_leaf      ↔  min_child_weight
  feature_fraction      ↔  colsample_bytree
  bagging_fraction      ↔  subsample
  lambda_l1/l2          ↔  reg_alpha/reg_lambda
  min_gain_to_split     ↔  gamma

PROS                         CONS
  ✅ Fastest training          ❌ Can overfit small datasets
  ✅ Low memory usage          ❌ More hyperparameters
  ✅ Native categoricals       ❌ leaf-wise harder to interpret
  ✅ EFB for sparse data       ❌ num_leaves != max_depth intuition
  ✅ GOSS for large data       ❌ Default params assume large n
═══════════════════════════════════════════════════════════
```
