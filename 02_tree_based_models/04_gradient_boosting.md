# Gradient Boosting — Complete Interview Guide

## 5-Pillar Answer Framework

| Pillar | Answer |
|--------|--------|
| **WHAT** | Gradient Boosting builds an additive ensemble of weak learners (typically shallow trees) by iteratively fitting each new tree to the **negative gradient** of the loss function with respect to the current ensemble's predictions — a generalisation of AdaBoost to arbitrary differentiable loss functions. |
| **WHY** | It achieves state-of-the-art accuracy on tabular data by combining gradient descent optimisation with the flexibility of tree ensembles. It can optimise virtually any differentiable loss function, making it universally applicable. |
| **WHEN** | Use on structured/tabular data when accuracy is the primary goal, data is clean, and training time is acceptable. It is the basis for the dominant algorithms in ML competitions (XGBoost, LightGBM, CatBoost). |
| **HOW** | Start with a constant prediction. At each round: compute pseudo-residuals (negative gradient of loss); fit a tree to these residuals; add the tree to the ensemble scaled by a learning rate. Repeat T times. |
| **PITFALLS** | Sequential training (cannot parallelise); requires careful tuning of learning rate and n_estimators; sensitive to outliers (use Huber or quantile loss instead of MSE/log-loss for robustness); slower than Random Forest. |

---

## Core Concept: Residual Fitting

The key insight of Gradient Boosting is that **fitting a new model to the residuals** of the current prediction is equivalent to performing gradient descent on the loss function in function space.

### Starting Simple: L2 Regression

Given current predictions `F(x)`, the residuals are:
```
rᵢ = yᵢ - F(xᵢ)
```

A new tree trained on these residuals, when added to F, reduces the MSE loss:
```
L = Σ (yᵢ - F(xᵢ))²

∂L/∂F(xᵢ) = -2(yᵢ - F(xᵢ)) = -2rᵢ

Pseudo-residual = -∂L/∂F(xᵢ) ∝ rᵢ  (the actual residual!)
```

So for MSE, pseudo-residuals **are** the ordinary residuals. This is why the concept is often introduced through residual fitting.

---

## Gradient Descent in Function Space

### Function-Space View

Instead of optimising parameters **w** directly (standard gradient descent), Gradient Boosting optimises the function **F(x)** itself:

```
Standard GD:  θₜ = θₜ₋₁ - η · ∂L/∂θ
              (move parameters in direction of steepest descent)

Functional GD: Fₜ(x) = Fₜ₋₁(x) - η · ∂L/∂F(x)
              (move the function in direction of steepest descent)
```

Since we can't represent arbitrary functions, we **approximate** the negative gradient with a tree hₜ(x):

```
Algorithm:
  F₀(x) = argmin_γ Σ L(yᵢ, γ)       ← initialise (e.g., mean of y)

  For t = 1, ..., T:
    rᵢₜ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]   ← pseudo-residuals (neg gradient)
    hₜ  = fit tree to {(xᵢ, rᵢₜ)}    ← tree approximates gradient direction
    γₜ  = argmin_γ Σ L(yᵢ, Fₜ₋₁(xᵢ) + γ·hₜ(xᵢ))   ← line search
    Fₜ(x) = Fₜ₋₁(x) + η · γₜ · hₜ(x)   ← update
```

---

## Pseudo-Residuals for Different Loss Functions

| Task | Loss Function | Formula | Pseudo-residual |
|------|--------------|---------|-----------------|
| Regression | MSE | `½(y - F)²` | `y - F` |
| Regression | MAE | `|y - F|` | `sign(y - F)` |
| Regression | Huber | Smooth combo of MSE+MAE | `y - F` if \|y-F\|≤δ, else `δ·sign(y-F)` |
| Binary Clf | Log Loss | `log(1+e^{-yF})` | `y - sigmoid(F)` |
| Multi-class | Softmax | `-Σ yₖ log pₖ` | `yₖ - pₖ` |
| Ranking | LambdaMART | NDCG-based | Pairwise gradient |

### Binary Classification Pseudo-Residuals (Key Derivation)

Loss: `L(y, F) = log(1 + e^{-yF})` where y ∈ {-1, +1}

```
∂L/∂F = -y · e^{-yF} / (1 + e^{-yF})
       = -y · (1 - sigmoid(yF))
       = -(y - p̂)      where p̂ = sigmoid(F)

Pseudo-residual: rᵢ = yᵢ - p̂ᵢ   ← difference between label and predicted probability
```

Gradient boosting for binary classification fits trees to probability residuals — exactly like fitting to residuals in regression, but in probability space.

---

## Learning Rate vs n_estimators Tradeoff

The learning rate η shrinks each tree's contribution:
```
F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₜ(x)
```

| Learning Rate | n_estimators | Effect |
|--------------|-------------|--------|
| 1.0 (high) | 100 | Fast convergence, risk of overfitting |
| 0.1 (medium) | 1000 | Better generalisation, standard practice |
| 0.01 (low) | 10000 | Very smooth, slow, but strong regulariser |
| 0.05 (low) | 2000 | Often optimal in competitions |

**Key principle (shrinkage):** Lower η + more trees usually outperforms higher η + fewer trees, at the cost of training time. This is one of the strongest regularisation strategies in boosting.

```
Intuition:
  Large η: each tree makes big jumps → overshoots optimum
  Small η: each tree makes small steps → more accurate direction,
           but need more steps to converge
```

**Rule of thumb:** Set η = 0.01–0.1, then find `n_estimators` via early stopping.

---

## Tree Structure Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `max_depth` | 3 | Depth of each tree (3–8 typical for GB) |
| `max_leaf_nodes` | None | Alternative to max_depth |
| `min_samples_leaf` | 1 | Minimum samples in leaves |
| `subsample` | 1.0 | Fraction of samples for each tree (Stochastic GB) |
| `max_features` | None | Random feature subsampling per split |

### Stochastic Gradient Boosting

When `subsample < 1.0`, each tree is trained on a random subsample — this:
- Reduces correlation between trees (like Random Forest)
- Acts as regularisation
- Speeds up training
- Often gives 10–20% accuracy improvement

---

## Complete Python Implementation

```python
import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_squared_error)
from sklearn.model_selection import GridSearchCV

# ── 1. Classification example ─────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Baseline GBM ───────────────────────────────────────────────────────────
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,           # stochastic GB
    min_samples_leaf=5,
    random_state=42
)
gbm.fit(X_train, y_train)

y_pred  = gbm.predict(X_test)
y_proba = gbm.predict_proba(X_test)[:, 1]

print("═" * 50)
print("GRADIENT BOOSTING CLASSIFIER")
print("═" * 50)
print(f"Test Accuracy: {gbm.score(X_test, y_test):.4f}")
print(f"ROC-AUC      : {roc_auc_score(y_test, y_proba):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── 3. Staged predict — find optimal n_estimators ────────────────────────────
print("Learning curve (staged predict):")
print(f"{'Round':>6} | {'Train AUC':>10} | {'Test AUC':>10}")
print("-" * 32)

staged_proba_train = list(gbm.staged_predict_proba(X_train))
staged_proba_test  = list(gbm.staged_predict_proba(X_test))

best_n, best_auc = 0, 0
for i, (tr_p, te_p) in enumerate(zip(staged_proba_train, staged_proba_test), 1):
    if i % 20 == 0 or i == 1:
        tr_auc = roc_auc_score(y_train, tr_p[:, 1])
        te_auc = roc_auc_score(y_test,  te_p[:, 1])
        print(f"{i:6d} | {tr_auc:10.4f} | {te_auc:10.4f}")
        if te_auc > best_auc:
            best_auc = te_auc
            best_n = i

print(f"\nOptimal n_estimators: {best_n}, best test AUC: {best_auc:.4f}")

# ── 4. Feature importance ─────────────────────────────────────────────────────
feature_names = list(data.feature_names)
importances   = gbm.feature_importances_
sorted_idx    = np.argsort(importances)[::-1][:10]

print("\nTop-10 Feature Importances:")
for rank, idx in enumerate(sorted_idx, 1):
    bar = "█" * int(importances[idx] * 300)
    print(f"  {rank:2d}. {feature_names[idx]:<35s} {importances[idx]:.4f} {bar}")

# ── 5. Hyperparameter grid search ─────────────────────────────────────────────
param_grid = {
    'n_estimators':  [100, 200, 400],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth':     [2, 3, 4],
    'subsample':     [0.7, 1.0],
}
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
best = grid_search.best_estimator_
print(f"\nBest params  : {grid_search.best_params_}")
print(f"Best CV AUC  : {grid_search.best_score_:.4f}")
print(f"Test ROC-AUC : {roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]):.4f}")

# ── 6. Regression example with Huber loss ────────────────────────────────────
X_reg, y_reg = make_regression(n_samples=500, n_features=10,
                                noise=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

for loss in ['squared_error', 'huber', 'absolute_error']:
    reg = GradientBoostingRegressor(
        loss=loss, n_estimators=200, learning_rate=0.1,
        max_depth=3, random_state=42
    )
    reg.fit(X_tr, y_tr)
    rmse = mean_squared_error(y_te, reg.predict(X_te)) ** 0.5
    print(f"Regression loss={loss:<20s}: RMSE = {rmse:.2f}")
```

---

## Interview Q&A

**Q1: What are pseudo-residuals and why are they called that?**

> For MSE loss, pseudo-residuals equal the ordinary residuals `yᵢ - Fₜ(xᵢ)`. For other loss functions, they are the **negative gradient** of the loss with respect to the current prediction — they "act like" residuals in a generalised sense, indicating the direction and magnitude in which the prediction should move. The name distinguishes them from true residuals when using non-MSE losses.

**Q2: How does learning rate act as regularisation?**

> By shrinking each tree's contribution by η, learning rate prevents any single tree from dominating the ensemble. It creates a smoother, more gradual optimisation trajectory in function space. Low η forces the model to take many small steps, reducing the chance of overfitting to individual training examples. Empirically, η = 0.01–0.1 with more estimators consistently outperforms η = 1.0 with fewer.

**Q3: What is the difference between Gradient Boosting and AdaBoost?**

> AdaBoost is a special case of Gradient Boosting where the loss is the **exponential loss** `e^{-yF}`. It uses sample reweighting as a mechanism equivalent to computing negative gradients for this specific loss. Gradient Boosting generalises the framework to any differentiable loss, uses direct gradient computation (pseudo-residuals) instead of reweighting, and supports regression, ranking, and custom objectives.

**Q4: Why is `max_depth=3–5` typical for Gradient Boosting (vs `None` for Random Forest)?**

> Random Forest uses deep trees to achieve low bias, relying on averaging to reduce variance. Gradient Boosting builds many shallow trees where each tree corrects specific errors of the ensemble. Shallow trees have high bias individually but low variance, and the boosting process reduces bias over rounds. Deeper trees in GB lead to faster overfitting. `max_depth=3` (with 8 leaves) is the classic choice.

**Q5: What is Stochastic Gradient Boosting?**

> When `subsample < 1.0`, each tree is trained on a random sample of training points (without replacement). This introduces stochasticity that: (1) decorrelates trees like Random Forest; (2) provides regularisation; (3) allows sublinear training time; (4) often improves generalisation by 10–20%. It is equivalent to Mini-batch SGD in neural networks.

**Q6: How do you prevent overfitting in Gradient Boosting?**

> Primary strategies: (1) Reduce `learning_rate` and increase `n_estimators` correspondingly; (2) Limit `max_depth` (3–5); (3) Increase `min_samples_leaf`; (4) Use `subsample < 1.0` (stochastic GB); (5) Use `max_features` for random split selection; (6) Early stopping via staged predict on a validation set. The most impactful: low learning rate + early stopping.

---

## Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| Overfitting | Too many estimators, high learning rate | Reduce lr; use early stopping |
| Slow training | Sequential, no parallelism within boost rounds | Use XGBoost/LightGBM instead |
| Sensitive to outliers (with MSE) | Large residuals dominate gradient | Use Huber or quantile loss |
| Ignoring subsample | Default is 1.0 (no stochasticity) | Set subsample=0.7–0.9 |
| Not tuning learning_rate + n_estimators together | Suboptimal performance | Always tune them jointly via early stopping |
| Forgetting feature scaling | Trees are scale-invariant | No scaling needed (unlike SVM, LR) |

---

## Quick Reference Cheat Sheet

```
GRADIENT BOOSTING — QUICK REFERENCE
═══════════════════════════════════════════════════════════

ALGORITHM
  F₀(x) = argmin_γ Σ L(yᵢ, γ)
  For t = 1 to T:
    rᵢₜ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)   ← pseudo-residuals
    Fit tree hₜ to {(xᵢ, rᵢₜ)}
    γₜ  = argmin_γ Σ L(yᵢ, Fₜ₋₁+γhₜ)
    Fₜ(x) = Fₜ₋₁(x) + η·γₜ·hₜ(x)

PSEUDO-RESIDUALS BY LOSS
  MSE      : rᵢ = yᵢ - F(xᵢ)
  Log Loss : rᵢ = yᵢ - sigmoid(F(xᵢ))
  Huber    : rᵢ = clip(yᵢ - F(xᵢ), -δ, δ)

KEY HYPERPARAMETERS
  n_estimators    → number of trees (tune with early stopping)
  learning_rate   → shrinkage (η); lower = better, but need more trees
  max_depth       → 3 (default), 3–8 typical for GB
  subsample       → 0.7–0.9 for stochastic GB
  min_samples_leaf → higher = smoother, less overfit
  max_features    → random split features (regularisation)

TYPICAL SKLEARN USAGE
  from sklearn.ensemble import GradientBoostingClassifier
  gbm = GradientBoostingClassifier(
      n_estimators=200, learning_rate=0.1,
      max_depth=3, subsample=0.8, random_state=42
  )

LOSS FUNCTIONS AVAILABLE
  Classification: 'log_loss' (default), 'exponential' (=AdaBoost)
  Regression    : 'squared_error', 'absolute_error', 'huber', 'quantile'

PROS                         CONS
  ✅ High accuracy             ❌ Sequential, slow to train
  ✅ Any differentiable loss   ❌ Many hyperparameters
  ✅ Handles mixed features    ❌ Sensitive to outliers (MSE)
  ✅ Feature importance        ❌ Needs careful regularisation
═══════════════════════════════════════════════════════════
```
