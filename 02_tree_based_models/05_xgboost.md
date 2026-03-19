# XGBoost — Complete Interview Guide

## 5-Pillar Answer Framework

| Pillar | Answer |
|--------|--------|
| **WHAT** | XGBoost (eXtreme Gradient Boosting) is an optimised, scalable gradient boosting framework that adds **L1/L2 regularisation** on tree leaf weights, **second-order Taylor expansion** of the loss, **column/row subsampling**, and sophisticated engineering optimisations (cache-aware computation, out-of-core learning, distributed training). |
| **WHY** | It achieves state-of-the-art accuracy on tabular data while training significantly faster than vanilla Gradient Boosting. The regularised objective prevents overfitting explicitly. Built-in handling of missing values, early stopping, and parallel split-finding make it production-ready. |
| **WHEN** | The default choice for structured/tabular ML problems in competitions and production. Use XGBoost when you need high accuracy, fast training, missing value handling, and a mature ecosystem with GPU support. |
| **HOW** | At each round, compute first-order gradients (gᵢ) and second-order gradients (hᵢ) of the loss. Find the split that maximises a regularised gain criterion derived from these gradients. Add the new tree (scaled by learning rate) to the ensemble. |
| **PITFALLS** | Still requires careful hyperparameter tuning; level-wise tree growth can waste computation on unnecessary nodes (LightGBM's leaf-wise growth is more efficient); very large `n_estimators` without early stopping leads to overfitting; many parameters create a large search space. |

---

## Regularised Objective Function

XGBoost adds an explicit regularisation term to the standard boosting objective:

```
Obj(θ) = Σᵢ L(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)

where:
  L(yᵢ, ŷᵢ)  = training loss (e.g., log-loss, MSE)
  Ω(fₜ)      = regularisation on tree t

  Ω(f) = γ·T + ½λ·Σⱼ wⱼ²  +  α·Σⱼ|wⱼ|
          └─────┘  └──────────┘    └──────────┘
          L0 on    L2 on leaves    L1 on leaves
          # leaves
```

Where:
- `T` = number of leaf nodes in the tree
- `wⱼ` = weight (predicted value) of leaf j
- `γ` = minimum gain required to make a split (L0 regularisation on tree complexity)
- `λ` = L2 regularisation on leaf weights (like Ridge)
- `α` = L1 regularisation on leaf weights (like Lasso)

**Effect of regularisation:**
- `γ > 0`: prevents splits that don't bring sufficient gain (like min_impurity_decrease)
- `λ > 0`: shrinks leaf weights toward zero, preventing extreme predictions
- `α > 0`: drives some leaf weights exactly to zero (sparse trees)

---

## Second-Order Taylor Expansion

The key mathematical innovation in XGBoost is using both the first and second derivatives of the loss:

### Taylor Expansion of Objective

At step t, the prediction is `ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)`.

Expand the loss around the current prediction:

```
L(yᵢ, ŷᵢ⁽ᵗ⁾) ≈ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + ½hᵢfₜ(xᵢ)²

where:
  gᵢ = ∂L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂ŷᵢ⁽ᵗ⁻¹⁾    (first-order gradient)
  hᵢ = ∂²L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂(ŷᵢ⁽ᵗ⁻¹⁾)²  (second-order gradient, Hessian)
```

### Optimal Leaf Weights

For a tree with leaf partition {Iⱼ}:

```
w*ⱼ = -(Σᵢ∈Iⱼ gᵢ) / (Σᵢ∈Iⱼ hᵢ + λ)
```

### Optimal Objective Value (Split Quality Score)

```
Obj* = -½ Σⱼ (Σᵢ∈Iⱼ gᵢ)² / (Σᵢ∈Iⱼ hᵢ + λ) + γT
```

### Gain from a Split

```
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ) ] - γ
        └──────────────┘  └──────────────┘  └──────────────────────────┘
           left child       right child          parent score
```

Where G = Σgᵢ (sum of gradients), H = Σhᵢ (sum of Hessians) in each region.

**Why second-order?**
- Vanilla GB uses only first-order info (steepest descent direction).
- XGBoost uses curvature (Hessian) to adaptively scale the step size — like Newton's method vs gradient descent.
- This leads to faster convergence and better leaf weight estimates.

### Gradients for Common Losses

| Loss | gᵢ (first) | hᵢ (second) |
|------|-----------|------------|
| MSE `½(y-ŷ)²` | `ŷ - y` | `1` |
| Log-loss | `p̂ - y` where p̂=sigmoid(ŷ) | `p̂(1-p̂)` |
| Softmax | `p̂ₖ - yₖ` | `p̂ₖ(1-p̂ₖ)` |
| Huber | clip(ŷ-y, -δ, δ) | `𝟙[|ŷ-y|≤δ]` |

---

## Subsampling: Column and Row

XGBoost supports multiple subsampling strategies:

### Row Subsampling (`subsample`)
```
subsample = 0.8  →  use 80% of training rows per tree
```
Same as Stochastic Gradient Boosting. Reduces variance, prevents overfitting.

### Column Subsampling
Three levels of column subsampling:

```
colsample_bytree  → sample features once per tree
colsample_bylevel → sample features once per depth level
colsample_bynode  → sample features once per split (like Random Forest)
```

Example:
```
p = 100 features, colsample_bytree=0.7, colsample_bynode=0.8
→ Each tree sees: 100×0.7 = 70 features
→ Each split sees: 70×0.8 = 56 features
```

These can be combined for aggressive regularisation on wide datasets.

---

## Missing Value Handling

XGBoost has built-in optimal missing value handling:

```
For each split candidate:
  1. Send all missing-value samples to the LEFT child
     → compute gain₁
  2. Send all missing-value samples to the RIGHT child
     → compute gain₂
  3. Choose direction with higher gain as the "default direction"
```

This learned direction is stored in the tree and used at inference time, making XGBoost naturally handle missing values without imputation.

---

## Early Stopping Implementation

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# ── 1. Data preparation ───────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ── 2. DMatrix format (XGBoost native) ───────────────────────────────────────
dtrain = xgb.DMatrix(X_tr,   label=y_tr)
dval   = xgb.DMatrix(X_val,  label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)

# ── 3. Training with early stopping ──────────────────────────────────────────
params = {
    'objective':         'binary:logistic',
    'eval_metric':       'auc',
    'learning_rate':     0.05,
    'max_depth':         4,
    'min_child_weight':  5,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,    # L1
    'reg_lambda':        1.0,    # L2
    'gamma':             0.0,    # min gain for split
    'seed':              42,
}

evals_result = {}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,        # stop if no improvement for 50 rounds
    evals_result=evals_result,
    verbose_eval=100
)

print(f"\nBest iteration: {bst.best_iteration}")
print(f"Best val AUC  : {bst.best_score:.4f}")
print(f"Test AUC      : {roc_auc_score(y_test, bst.predict(dtest)):.4f}")

# ── 4. sklearn API with early stopping ────────────────────────────────────────
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"\nsklearn API — Best iteration: {xgb_clf.best_iteration}")
print(f"Test AUC: {roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1]):.4f}")

# ── 5. Feature importance ─────────────────────────────────────────────────────
importance_types = ['weight', 'gain', 'cover']
for imp_type in importance_types:
    scores = bst.get_score(importance_type=imp_type)
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\nTop-3 features ({imp_type}): {top3}")

# ── 6. Cross-validation with xgb.cv ──────────────────────────────────────────
dall = xgb.DMatrix(X_train, label=y_train)
cv_results = xgb.cv(
    params,
    dall,
    num_boost_round=500,
    nfold=5,
    metrics=['auc'],
    early_stopping_rounds=30,
    seed=42,
    verbose_eval=False
)

best_n = cv_results['test-auc-mean'].idxmax() + 1
best_auc = cv_results['test-auc-mean'].max()
print(f"\nCV best n_estimators: {best_n}, AUC: {best_auc:.4f}")
```

---

## Complete Hyperparameter Guide with Typical Ranges

### Core Parameters

| Parameter | Default | Typical Range | Description |
|-----------|---------|--------------|-------------|
| `n_estimators` | 100 | 100–5000 | Number of boosting rounds; tune with early stopping |
| `learning_rate` (`eta`) | 0.3 | 0.01–0.3 | Shrinkage; lower = more robust |
| `max_depth` | 6 | 3–10 | Max tree depth; deeper = more complex |
| `min_child_weight` | 1 | 1–10 | Min sum of Hessian in child (=min_samples_leaf analogue) |

### Regularisation Parameters

| Parameter | Default | Typical Range | Description |
|-----------|---------|--------------|-------------|
| `gamma` (`min_split_loss`) | 0 | 0–5 | Min gain to make a split |
| `reg_alpha` | 0 | 0–10 | L1 regularisation on weights |
| `reg_lambda` | 1 | 0.1–10 | L2 regularisation on weights |
| `max_delta_step` | 0 | 0–10 | Limits max delta step per step (for imbalanced classification) |

### Subsampling Parameters

| Parameter | Default | Typical Range | Description |
|-----------|---------|--------------|-------------|
| `subsample` | 1.0 | 0.5–1.0 | Row subsampling per tree |
| `colsample_bytree` | 1.0 | 0.3–1.0 | Column subsampling per tree |
| `colsample_bylevel` | 1.0 | 0.5–1.0 | Column subsampling per level |
| `colsample_bynode` | 1.0 | 0.5–1.0 | Column subsampling per split |

### Recommended Starting Point

```python
params = {
    'learning_rate':    0.05,
    'max_depth':        4,
    'min_child_weight': 5,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'n_estimators':     2000,    # use early stopping
    'early_stopping_rounds': 50,
}
```

---

## Interview Q&A

**Q1: What makes XGBoost faster than vanilla sklearn GradientBoosting?**

> Multiple engineering improvements: (1) **Parallel split-finding** — XGBoost finds the best split by sorting feature values once and scanning thresholds in parallel across features; (2) **Cache-aware access** — data laid out for cache efficiency; (3) **Approximate tree learning** — uses quantile sketches to find split candidates instead of exact enumeration; (4) **Out-of-core computation** — handles datasets larger than RAM; (5) **GPU support** — can train on GPU for massive speedups.

**Q2: How does the second-order Taylor expansion improve XGBoost?**

> Standard Gradient Boosting uses only first-order gradients (steepest descent direction). XGBoost uses both first (gᵢ) and second (hᵢ) derivatives, analogous to Newton's method vs gradient descent. The Hessian provides curvature information, enabling more accurate step sizing and optimal leaf weight computation: `w* = -G/(H+λ)`. This leads to faster convergence and better solutions.

**Q3: What is `min_child_weight` and how does it regularise?**

> `min_child_weight` is the minimum sum of Hessians (`Σhᵢ`) required in a leaf node. For log-loss, `hᵢ = pᵢ(1-pᵢ)`, so this is roughly proportional to minimum samples (when predictions are ~0.5, hᵢ ≈ 0.25). It prevents splits that would create nodes with very few effective samples, acting as a stronger regulariser than simple `min_samples_leaf` because it's weighted by prediction uncertainty.

**Q4: How does XGBoost handle missing values?**

> During training, XGBoost learns the optimal **default direction** (left or right) for missing values at each split — it tries both directions and picks the one that gives higher gain. At inference, missing values are routed in the learned default direction. This is superior to imputation because the direction is optimised for the specific split and loss function.

**Q5: Explain the `gamma` parameter.**

> `gamma` (also called `min_split_loss`) is the minimum gain a split must achieve to be accepted. From the gain formula: `Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ`. A split is only made if `Gain > 0`, i.e., if the improvement exceeds γ. Higher γ → more conservative splits → simpler trees. It's analogous to `min_impurity_decrease` in sklearn but operates on the regularised XGBoost objective.

**Q6: What's the difference between `weight`, `gain`, and `cover` feature importances in XGBoost?**

> `weight` = number of times the feature is used as a split (similar to sklearn MDI count). `gain` = average gain of splits using that feature (weighted by how much it improves the objective — more informative than weight). `cover` = average number of samples covered by splits using that feature. `gain` is generally the most informative importance type for understanding actual predictive contribution.

---

## Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| Overfitting without early stopping | Default n_estimators may be too low/high | Always use early stopping + eval_set |
| High learning_rate with default n_estimators | Fast convergence to suboptimal solution | Set lr=0.05, n_estimators=2000, use early stopping |
| Ignoring scale_pos_weight | Class imbalance | Set `scale_pos_weight = neg/pos count` |
| Using 'weight' importance for feature selection | Biased, doesn't reflect predictive value | Use 'gain' importance or SHAP values |
| Not tuning min_child_weight | Controls leaf minimum samples | Tune with grid search (1–20) |
| colsample_bytree=1.0 on wide data | Overfitting on correlated features | Set 0.5–0.8 |

---

## Quick Reference Cheat Sheet

```
XGBOOST — QUICK REFERENCE
═══════════════════════════════════════════════════════════

OBJECTIVE
  Obj = Σ L(yᵢ,ŷᵢ) + Σₜ [γT + ½λΣwⱼ² + αΣ|wⱼ|]

OPTIMAL LEAF WEIGHTS
  w*ⱼ = -(Σᵢ∈leaf gᵢ) / (Σᵢ∈leaf hᵢ + λ)

SPLIT GAIN
  Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ

KEY HYPERPARAMETERS
  Core:         n_estimators, learning_rate, max_depth
  Child:        min_child_weight (1–10)
  Regularise:   gamma, reg_alpha, reg_lambda
  Subsample:    subsample (0.8), colsample_bytree (0.8)
  Imbalance:    scale_pos_weight = neg_count/pos_count

MISSING VALUES: Learned default direction per split (no imputation needed)

EARLY STOPPING
  xgb.train(..., early_stopping_rounds=50,
            evals=[(dval,'val')], ...)

TYPICAL STARTING PARAMS
  learning_rate=0.05, max_depth=4, min_child_weight=5,
  subsample=0.8, colsample_bytree=0.8, n_estimators=2000

FEATURE IMPORTANCE TYPES
  'weight' → split count (biased)
  'gain'   → avg objective gain (recommended)
  'cover'  → avg sample coverage

PROS                         CONS
  ✅ Regularised objective     ❌ Many hyperparameters
  ✅ Built-in missing handling ❌ Slower than LightGBM
  ✅ Second-order optimisation ❌ Level-wise growth
  ✅ Early stopping built-in   ❌ Memory intensive
  ✅ GPU support               ❌ Less efficient on sparse data
═══════════════════════════════════════════════════════════
```
