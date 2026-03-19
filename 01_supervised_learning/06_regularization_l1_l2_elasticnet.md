# 🎛️ Regularization: L1, L2, and ElasticNet — Complete Interview Guide

> **5-Pillar Answer Framework:** WHAT | WHY | WHEN | HOW | PITFALLS

---

## 🏛️ The 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | Regularization adds a penalty term to the loss function that discourages large weights, preventing overfitting by controlling model complexity |
| **WHY** | Without regularization, models overfit by memorizing training noise; regularization trades a small increase in bias for a large reduction in variance |
| **WHEN** | Many features (p > n), correlated features, training error << validation error, need feature selection (L1), or robust handling of correlation (L2) |
| **HOW** | L1 adds λΣ\|βⱼ\|, L2 adds λΣβⱼ², ElasticNet adds both; larger λ → more regularization → simpler model |
| **PITFALLS** | Too much regularization → underfitting; not scaling features → different features penalized differently; forgetting to tune λ |

---

## 📐 Mathematical Formulations

### Ridge Regression (L2)
```
L_Ridge(β) = RSS + λ Σⱼ βⱼ²
           = ||y - Xβ||² + λ||β||²₂

Closed-form solution:  β_Ridge = (XᵀX + λI)⁻¹ Xᵀy
```

**Key insight:** Adding λI to XᵀX makes it invertible even when XᵀX is singular (multicollinearity). Ridge **always** has a unique solution.

### Lasso Regression (L1)
```
L_Lasso(β) = RSS + λ Σⱼ |βⱼ|
           = ||y - Xβ||² + λ||β||₁

No closed-form (non-differentiable at βⱼ = 0)
Solved by: Coordinate Descent, LARS (Least Angle Regression)
```

**Key insight:** L1 penalty creates exact zeros → automatic feature selection.

### ElasticNet (L1 + L2)
```
L_EN(β) = RSS + λ[ρ||β||₁ + (1-ρ)||β||²₂]

         = RSS + α₁ Σⱼ|βⱼ| + α₂ Σⱼβⱼ²
```

Where ρ (l1_ratio in sklearn) ∈ [0, 1] balances L1 and L2.
- ρ = 0 → Pure Ridge
- ρ = 1 → Pure Lasso
- 0 < ρ < 1 → ElasticNet

---

## �� Geometric Interpretation

### Why L1 Creates Sparsity (Corners of the Diamond)

```
Constraint Region       Loss Function (RSS) Contours
────────────────────────────────────────────────────
L2 (Ridge): Circle      ○  Smooth, no corners
                           Solution lands anywhere on circle
                           Coefficients shrink but ≠ 0

L1 (Lasso): Diamond     ◇  Has corners on axes (β₁=0 or β₂=0)
                           Elliptical loss contours hit corners first
                           Forces some βⱼ = 0 exactly
```

**Formal explanation:**
- OLS loss contours are ellipses centered at β_OLS
- The first point of contact between shrinking ellipse and constraint region is the solution
- Ellipse hits the **corner** of the L1 diamond (where one coordinate = 0) far more often than hitting a point on the L2 circle boundary
- This is why L1 produces sparse solutions and L2 doesn't

### Bayesian Interpretation

| Regularization | Equivalent Prior on β | Distribution |
|----------------|----------------------|--------------|
| None (OLS) | Flat/improper prior | Uniform |
| Ridge (L2) | Gaussian prior: β ~ N(0, σ²/λ) | Gaussian → smooth |
| Lasso (L1) | Laplace (double-exp) prior: β ~ Laplace(0, 1/λ) | Heavy tails → zeros |
| ElasticNet | Mixture prior | Gaussian + Laplace |

---

## 📊 Why L2 Handles Correlated Features

For two perfectly correlated features x₁ ≡ x₂:
- **OLS**: Infinite solutions — coefficient can be split arbitrarily between x₁ and x₂
- **Ridge**: Unique solution — splits weight equally between correlated features (β₁ = β₂)
- **Lasso**: Selects one arbitrarily — keeps either x₁ or x₂ but not both

**In practice:** Lasso with correlated groups selects one representative randomly. Ridge keeps all features but shrinks equally. ElasticNet selects the group while shrinking within it.

---

## 📈 Regularization Path

As λ increases from 0 to ∞:
- **Ridge**: All coefficients continuously shrink toward 0, never reaching exactly 0
- **Lasso**: Coefficients shrink and sequentially hit exactly 0; features enter model in order of correlation with residuals
- **ElasticNet**: Combination — some zeros (like Lasso) + equal shrinkage of correlated groups (like Ridge)

---

## 💻 Complete Implementation with Alpha/Lambda Selection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV,
    ElasticNet, ElasticNetCV,
    lars_path, lasso_path
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, KFold, train_test_split
)
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ── Dataset: many features, some informative ─────────────────────
X, y, true_coefs = make_regression(
    n_samples=300,
    n_features=50,       # 50 features
    n_informative=15,    # only 15 truly informative
    noise=20,
    coef=True,           # Return true coefficients
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"True informative features: {np.sum(true_coefs != 0)}/50")


# ══════════════════════════════════════════════════════════════════
# PART 1: RIDGE (L2)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("RIDGE REGRESSION")
print("=" * 55)

# ── Auto alpha selection via RidgeCV ─────────────────────────────
alphas = np.logspace(-3, 5, 200)

ridge_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RidgeCV(
        alphas=alphas,
        cv=10,           # 10-fold CV
        scoring='r2',
        fit_intercept=True
    ))
])
ridge_cv.fit(X_train, y_train)
best_alpha_ridge = ridge_cv.named_steps['model'].alpha_
ridge_coefs = ridge_cv.named_steps['model'].coef_

print(f"Best alpha (Ridge): {best_alpha_ridge:.4f}")
print(f"Train R²: {ridge_cv.score(X_train, y_train):.4f}")
print(f"Test  R²: {ridge_cv.score(X_test, y_test):.4f}")
print(f"Non-zero coefs: {np.sum(ridge_coefs != 0)}/50 (Ridge keeps all)")

# ── Manual Ridge: coefficient shrinkage analysis ─────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

test_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge_coef_paths = []
for a in test_alphas:
    ridge = Ridge(alpha=a, fit_intercept=True)
    ridge.fit(X_train_s, y_train)
    ridge_coef_paths.append(ridge.coef_)

print("\nRidge: Coefficient magnitude shrinks with alpha:")
print(f"{'Alpha':<10} {'Max |coef|':>12} {'Mean |coef|':>12} {'Test R²':>10}")
for a, coefs in zip(test_alphas, ridge_coef_paths):
    ridge_temp = Ridge(alpha=a)
    ridge_temp.fit(X_train_s, y_train)
    test_r2 = ridge_temp.score(X_test_s, y_test)
    print(f"{a:<10.3f} {np.max(np.abs(coefs)):>12.4f} {np.mean(np.abs(coefs)):>12.4f} {test_r2:>10.4f}")


# ══════════════════════════════════════════════════════════════════
# PART 2: LASSO (L1)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("LASSO REGRESSION")
print("=" * 55)

# ── LassoCV: automatic alpha selection ───────────────────────────
lasso_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LassoCV(
        alphas=alphas,
        cv=10,
        max_iter=10000,
        tol=1e-4,
        random_state=42
    ))
])
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.named_steps['model'].alpha_
lasso_coefs = lasso_cv.named_steps['model'].coef_
n_selected = np.sum(lasso_coefs != 0)

print(f"Best alpha (Lasso): {best_alpha_lasso:.6f}")
print(f"Train R²: {lasso_cv.score(X_train, y_train):.4f}")
print(f"Test  R²: {lasso_cv.score(X_test, y_test):.4f}")
print(f"Features selected: {n_selected}/50 (others set to exactly 0)")
print(f"Selected feature indices: {np.where(lasso_coefs != 0)[0].tolist()}")

# ── Lasso: sparsity increases with alpha ─────────────────────────
print("\nLasso: Sparsity increases with alpha:")
print(f"{'Alpha':<12} {'Non-zero':>10} {'Test R²':>10}")
for a in test_alphas:
    lasso_temp = Lasso(alpha=a, max_iter=10000)
    lasso_temp.fit(X_train_s, y_train)
    nz = np.sum(lasso_temp.coef_ != 0)
    test_r2 = lasso_temp.score(X_test_s, y_test)
    print(f"{a:<12.3f} {nz:>10}/50 {test_r2:>10.4f}")


# ══════════════════════════════════════════════════════════════════
# PART 3: ELASTICNET
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("ELASTICNET REGRESSION")
print("=" * 55)

en_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ElasticNetCV(
        l1_ratio=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 2, 50),
        cv=10,
        max_iter=10000,
        random_state=42
    ))
])
en_cv.fit(X_train, y_train)
en_model = en_cv.named_steps['model']
en_coefs = en_model.coef_

print(f"Best alpha:    {en_model.alpha_:.6f}")
print(f"Best l1_ratio: {en_model.l1_ratio_:.2f}")
print(f"Train R²: {en_cv.score(X_train, y_train):.4f}")
print(f"Test  R²: {en_cv.score(X_test, y_test):.4f}")
print(f"Non-zero coefs: {np.sum(en_coefs != 0)}/50")

# ── l1_ratio Effect ───────────────────────────────────────────────
print("\nElasticNet: Effect of l1_ratio (alpha=0.1):")
print(f"{'l1_ratio':<12} {'Non-zero':>10} {'Test R²':>10}")
for l1r in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    en = ElasticNet(alpha=0.1, l1_ratio=l1r, max_iter=10000)
    en.fit(X_train_s, y_train)
    nz = np.sum(en.coef_ != 0)
    test_r2 = en.score(X_test_s, y_test)
    label = "(Ridge)" if l1r == 0 else ("(Lasso)" if l1r == 1.0 else "")
    print(f"{l1r:<12.1f} {nz:>10}/50 {test_r2:>10.4f} {label}")


# ══════════════════════════════════════════════════════════════════
# PART 4: REGULARIZATION PATH VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_regularization_paths(X_train, y_train):
    """Plot how coefficients change with regularization strength."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    alphas_ridge = np.logspace(-2, 4, 100)
    alphas_lasso = np.logspace(-4, 1, 100)

    # Ridge path
    coefs_ridge = []
    for a in alphas_ridge:
        r = Ridge(alpha=a)
        r.fit(X_s, y_train)
        coefs_ridge.append(r.coef_)

    # Lasso path
    coefs_lasso = []
    for a in alphas_lasso:
        l = Lasso(alpha=a, max_iter=5000)
        l.fit(X_s, y_train)
        coefs_lasso.append(l.coef_)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Ridge plot
    axes[0].semilogx(alphas_ridge, coefs_ridge)
    axes[0].set_xlabel('Alpha (λ)')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].set_title('Ridge (L2): Coefficient Path\n(All shrink, none reach exactly 0)')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].grid(alpha=0.3)

    # Lasso plot
    axes[1].semilogx(alphas_lasso, coefs_lasso)
    axes[1].set_xlabel('Alpha (λ)')
    axes[1].set_ylabel('Coefficient Value')
    axes[1].set_title('Lasso (L1): Coefficient Path\n(Features sequentially set to exactly 0)')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('regularization_paths.png', dpi=100)
    print("Regularization path plot saved.")

plot_regularization_paths(X_train, y_train)


# ══════════════════════════════════════════════════════════════════
# PART 5: COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

from sklearn.linear_model import LinearRegression
models = {
    'OLS (no reg)': Pipeline([('s', StandardScaler()), ('m', LinearRegression())]),
    'Ridge (L2)':   ridge_cv,
    'Lasso (L1)':   lasso_cv,
    'ElasticNet':   en_cv,
}

print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Non-zero coefs':>16}")
print("-" * 60)
for name, model in models.items():
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    if hasattr(model.named_steps.get('model', None), 'coef_'):
        nz = np.sum(model.named_steps['model'].coef_ != 0)
    else:
        nz = 50
    print(f"{name:<20} {train_r2:>10.4f} {test_r2:>10.4f} {nz:>16}/50")
```

---

## ❓ Interview Questions & Answers

### Q1: Geometrically, why does L1 create sparsity but L2 doesn't?
**Answer:** The OLS solution sits at the minimum of the RSS ellipses. Regularization adds a constraint: L1 constrains to a diamond (|β₁| + |β₂| ≤ t), L2 to a circle (β₁² + β₂² ≤ t). The elliptical contours of RSS expand from the OLS minimum until they touch the constraint region. The diamond has **corners on the axes** (where β₁=0 or β₂=0), and the probability of the ellipse hitting a corner is much higher than landing on the smooth circle boundary. Hence L1 forces exact zeros.

### Q2: Why is Ridge preferred when features are correlated?
**Answer:** With correlated features, OLS has high variance (XᵀX nearly singular → large (XᵀX)⁻¹). Ridge adds λI making β_Ridge = (XᵀX + λI)⁻¹Xᵀy — always invertible. For perfectly correlated x₁ ≡ x₂, Ridge gives equal coefficients (β₁ = β₂), distributing weight stably. Lasso arbitrarily picks one correlated feature and drops others, leading to unstable feature selection (different random seeds → different features selected).

### Q3: What is the difference between lambda (λ) and alpha in sklearn?
**Answer:** They're the same concept but named differently. Mathematically λ and α both refer to the regularization strength. In sklearn: `Ridge(alpha=λ)`, `Lasso(alpha=λ)` — larger alpha means more regularization. In `LogisticRegression`, C = 1/λ — larger C means less regularization. Always clarify convention when discussing regularization strength. The key is: more regularization = simpler model = more bias, less variance.

### Q4: How do you select the optimal regularization strength?
**Answer:** Use **cross-validation**: `RidgeCV(alphas=np.logspace(-3, 5, 100))` tests a range of alphas on held-out folds. Search on log scale (0.001, 0.01, 0.1, 1, 10, 100, 1000) — regularization effect is approximately log-linear. Always scale features first, otherwise different features need different λ values. For ElasticNet, tune both alpha and l1_ratio jointly with `ElasticNetCV`.

### Q5: When should you use ElasticNet over pure Lasso or Ridge?
**Answer:** Use **ElasticNet** when: (1) You have groups of correlated features and want to select the group (Lasso picks arbitrarily, Ridge keeps all), (2) p >> n (more features than samples) — Lasso selects at most n features, ElasticNet can select more, (3) You want sparsity but stability of Ridge. Rule of thumb: if unsure between L1 and L2, start with ElasticNet and let CV choose l1_ratio.

---

## 📊 L1 vs L2 vs ElasticNet Comparison

| Property | L1 (Lasso) | L2 (Ridge) | ElasticNet |
|----------|-----------|-----------|-----------|
| **Sparsity** | ✅ Yes (exact zeros) | ❌ No (shrinks toward 0) | ✅ Yes (fewer zeros than L1) |
| **Feature selection** | ✅ Automatic | ❌ Keeps all | ✅ Yes |
| **Correlated features** | ❌ Picks one arbitrarily | ✅ Groups together | ✅ Groups + selects |
| **p > n** | ✅ Selects at most n | ✅ Works but keeps all | ✅ Selects more than n |
| **Unique solution** | ❌ Not always | ✅ Always | ✅ Always |
| **Closed form** | ❌ No | ✅ Yes | ❌ No |
| **Solver** | Coordinate descent | Cholesky/SVD | Coordinate descent |
| **Interpretability** | ✅ Sparse, easy to read | ⚠️ All non-zero | ✅ Sparse + stable |

---

## ⚠️ Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Not scaling features | Features penalized unevenly | StandardScaler before regularization |
| Too large λ | High train AND test error (underfitting) | Reduce λ, use CV |
| Too small λ | Low train error, high test error (overfitting) | Increase λ, use CV |
| Using Lasso with correlated features | Unstable feature selection | Use ElasticNet |
| Not searching on log scale | Miss optimal λ | Use np.logspace(-3, 5, 100) |
| Regularizing the intercept | Biased intercept | `fit_intercept=True` (default, not regularized) |

---

## 📋 Quick Reference Cheat Sheet

```
Regularization Quick Reference
══════════════════════════════════════════════════════════
Ridge (L2):    L = RSS + λ||β||²₂   → Shrinks all, no zeros
Lasso (L1):    L = RSS + λ||β||₁    → Exact zeros, feature selection
ElasticNet:    L = RSS + λ[ρ||β||₁ + (1-ρ)||β||²₂]

Geometry:
  L2 constraint: circle → no corners → no exact zeros
  L1 constraint: diamond → corners on axes → exact zeros

Bayesian:
  L2 ↔ Gaussian prior on β
  L1 ↔ Laplace prior on β

When to use:
  Ridge:      Correlated features, keep all features, p < n
  Lasso:      Feature selection needed, sparse solution
  ElasticNet: Grouped correlated features, p >> n

Always scale features! (StandardScaler)
Tune on log scale: alphas = np.logspace(-3, 5, 100)

sklearn:
  RidgeCV(alphas=..., cv=10)
  LassoCV(alphas=..., cv=10, max_iter=10000)
  ElasticNetCV(l1_ratio=[.1,.5,.9,.95,1], cv=10)

Note: sklearn's 'alpha' = mathematical λ (regularization strength)
      sklearn LogisticRegression's 'C' = 1/λ (inverse!)
══════════════════════════════════════════════════════════
```
