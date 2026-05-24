# ⚔️ Support Vector Machines — Complete Interview Guide

> **5-Pillar Answer Framework:** WHAT | WHY | WHEN | HOW | PITFALLS

---

## 🏛️ The 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | A discriminative classifier that finds the maximum-margin hyperplane separating classes; uses kernel trick for non-linear boundaries |
| **WHY** | Effective in high dimensions, memory efficient (uses only support vectors), theoretically grounded via VC dimension, versatile via kernels |
| **WHEN** | High-dimensional data (text, images), small-to-medium datasets, clear margin of separation, non-linear boundaries (with RBF kernel) |
| **HOW** | Solves quadratic programming problem; C controls soft-margin; kernel K(x,z) replaces dot products for non-linear mapping |
| **PITFALLS** | Slow on large datasets O(n²-n³), no probability output by default, sensitive to feature scaling, hard to interpret |

---

## 📐 Mathematical Foundation

### Hard-Margin SVM (Linearly Separable)

Find hyperplane w·x + b = 0 maximizing the margin 2/||w||:

**Primal Formulation:**
```
Minimize:    1/2 ||w||²
Subject to:  yᵢ(w·xᵢ + b) ≥ 1   for all i
```

**Support Vectors** = training points on the margin boundary (yᵢ(w·xᵢ + b) = 1).
Only support vectors affect the decision boundary — other points can be moved without changing it.

### Soft-Margin SVM (C Parameter)

Real data is rarely linearly separable. Introduce slack variables ξᵢ ≥ 0:

```
Minimize:    1/2 ||w||² + C Σᵢ ξᵢ
Subject to:  yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
             ξᵢ ≥ 0
```

**C parameter:**
- Large C → small margin, fewer misclassifications → risk of overfitting
- Small C → large margin, more misclassifications tolerated → better generalization
- C → ∞: hard-margin SVM

### Dual Formulation & Kernel Trick

The dual problem depends only on dot products x·z:
```
Maximize:    Σᵢ αᵢ - 1/2 Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)
Subject to:  Σᵢ αᵢ yᵢ = 0,  0 ≤ αᵢ ≤ C
```

Decision function:  f(x) = Σᵢ αᵢ yᵢ (xᵢ·x) + b

Replace (xᵢ·x) with **K(xᵢ, x)** → implicit high-dim mapping!

### Common Kernels

| Kernel | Formula | Use Case | Hyperparameters |
|--------|---------|----------|-----------------|
| **Linear** | K(x,z) = xᵀz | High-dim, text | None |
| **RBF (Gaussian)** | K(x,z) = exp(-γ‖x-z‖²) | General purpose | γ |
| **Polynomial** | K(x,z) = (xᵀz + r)^d | Computer vision | d, r |
| **Sigmoid** | K(x,z) = tanh(κxᵀz + θ) | Neural net-like | κ, θ |

**RBF γ parameter:**
- Large γ → each point has small influence → complex boundary → overfitting
- Small γ → smooth boundary → underfitting

---

## 🔄 SVC vs SVR Comparison

| Aspect | SVC (Classification) | SVR (Regression) |
|--------|---------------------|-----------------|
| **Goal** | Maximum margin between classes | Fit data within ε-tube |
| **Loss** | Hinge loss | ε-insensitive loss |
| **Output** | Class label or score | Continuous value |
| **Key param** | C (soft-margin) | C, ε (tube width) |
| **Support vectors** | Points on/inside margin | Points outside ε-tube |

**SVR loss:** L(y, f(x)) = max(0, |y - f(x)| - ε)

---

## 🌳 SVM vs Tree Models

| Criterion | SVM | Random Forest / XGBoost |
|-----------|-----|------------------------|
| Dataset size | Small-medium (< 100K) | Any (millions+) |
| Training speed | Slow O(n²-n³) | Fast O(n log n) |
| Inference speed | Fast (few SVs) | Fast |
| Missing values | Must impute | Handles natively |
| Feature scaling | Required | Not needed |
| Interpretability | Low | Medium (feature importance) |
| Non-linear data | Kernels (RBF) | Native |
| Probability output | Platt scaling (slower) | Direct |
| Hyperparameters | C, γ | Many (n_estimators, depth, etc.) |
| Best for | High-dim text, small datasets | Tabular data, large datasets |

---

## 💻 Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
)
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ── Dataset ───────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=1000, n_features=15, n_informative=10,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 1. Linear SVC (fast for large datasets) ───────────────────────
linear_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])
linear_svc.fit(X_train, y_train)
print(f"LinearSVC Accuracy: {linear_svc.score(X_test, y_test):.4f}")

# ── 2. SVC with RBF Kernel ────────────────────────────────────────
rbf_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',       # gamma = 1/(n_features * X.var()) — recommended
        probability=True,    # Enable predict_proba (uses Platt scaling)
        random_state=42,
        class_weight=None    # Set 'balanced' for imbalanced data
    ))
])
rbf_svc.fit(X_train, y_train)
y_proba = rbf_svc.predict_proba(X_test)[:, 1]
print(f"\nRBF SVC ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, rbf_svc.predict(X_test)))

# ── 3. Grid Search: Tune C and gamma ─────────────────────────────
param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(
    estimator=Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', probability=True))
    ]),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]):.4f}")

# ── 4. SVR (Regression) ───────────────────────────────────────────
X_r, y_r = make_regression(n_samples=500, n_features=10, noise=15, random_state=42)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42
)

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR(
        kernel='rbf',
        C=10.0,       # Penalty for points outside ε-tube
        epsilon=0.1,  # Width of the insensitive tube
        gamma='scale'
    ))
])
svr_pipeline.fit(X_r_train, y_r_train)
print(f"\nSVR R²: {svr_pipeline.score(X_r_test, y_r_test):.4f}")

# ── 5. Decision Boundary Visualization (2D) ───────────────────────
def plot_svm_decision_boundary(X, y, svm_model, title="SVM Decision Boundary"):
    """Visualize SVM decision boundary and support vectors."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], alpha=0.3, colors=['#FFAAAA', '#AAAAFF'])
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                linestyles=['--', '-', '--'])
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', alpha=0.6, edgecolors='k')
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='green', linewidths=2,
                label=f'Support Vectors ({len(svm.support_vectors_)})')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig('svm_boundary.png', dpi=100)
    print(f"Number of support vectors: {len(svm.support_vectors_)}")
```

---

## ❓ Interview Questions & Answers

### Q1: What are support vectors and why do only they matter?
**Answer:** Support vectors are the training points closest to the decision hyperplane — on the margin boundaries. The decision function f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b has non-zero αᵢ **only** for support vectors (KKT conditions: αᵢ = 0 for points correctly classified with margin > 1). Moving other points doesn't change the boundary. Typically only 10-30% of training points are SVs.

### Q2: Explain the kernel trick — why is it computationally efficient?
**Answer:** Without kernels, mapping to d-dimensional space and computing dot products costs O(d). The kernel trick computes K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) **without** explicitly computing φ(x). For RBF kernel, the implicit feature space is infinite-dimensional, yet K(x,z) = exp(-γ‖x-z‖²) takes O(p) to compute. We only need the n×n Gram matrix K.

### Q3: How do you choose between linear and RBF kernels?
**Answer:** Start with **linear** if: p >> n (text, gene data), or you expect a linear boundary. Use **RBF** if: n >> p and the boundary is non-linear. Rule of thumb: if LinearSVC works well, no need for RBF. Always scale features first. For very large n, use LinearSVC (scales to millions) since RBF SVC is O(n²-n³).

### Q4: Why must features be scaled for SVMs?
**Answer:** SVM maximizes margin in Euclidean space: ||w||². If features have different scales, the distance metric is dominated by large-scale features. A feature with values [0, 1000] will dominate one with values [0, 1]. StandardScaler ensures all features contribute equally to the margin calculation. Unlike tree models, SVMs are highly sensitive to scale.

### Q5: What does the C parameter control in SVM?
**Answer:** C is the penalty for margin violations. **High C**: penalize misclassifications heavily → small margin, complex boundary → potential overfitting. **Low C**: allow more misclassifications → large margin → better generalization. C is inversely related to regularization strength (like 1/λ). Tune via GridSearchCV on log scale: [0.001, 0.01, 0.1, 1, 10, 100].

---

## ⚠️ Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Not scaling features | Poor performance, slow convergence | Always apply StandardScaler |
| Using RBF on large datasets | Training takes hours | Use LinearSVC or kernel approximation (Nystroem) |
| Forgetting probability=True | Can't call predict_proba | Set `probability=True` in SVC (slower due to Platt scaling) |
| Not tuning C and gamma together | Suboptimal performance | Always GridSearch both jointly |
| Using SVM for feature importance | Not directly available | Use permutation importance or switch to tree model |
| Large C with noisy data | Overfitting | Reduce C to allow soft margin |

---

## 📋 Quick Reference Cheat Sheet

```
SVM Quick Reference
══════════════════════════════════════════════════════════
Hard Margin:    min 1/2||w||²  s.t. yᵢ(w·xᵢ+b) ≥ 1
Soft Margin:    min 1/2||w||² + C·Σξᵢ  s.t. yᵢ(wᵀxᵢ+b) ≥ 1-ξᵢ
Kernel:         K(x,z) replaces dot products → implicit high-dim mapping

Kernels:
  Linear:   K = xᵀz           (use for text, high-dim)
  RBF:      K = exp(-γ||x-z||²)  (general purpose)
  Poly:     K = (xᵀz + r)^d   (computer vision)

C parameter: large C → complex, small C → simple (regularized)
γ parameter: large γ → complex, small γ → smooth

SVC:  Classification, hinge loss
SVR:  Regression, ε-insensitive loss (C, ε parameters)
LinearSVC: Fast for large datasets, no kernel

Must scale features! Use StandardScaler.
Tune: GridSearchCV on C=[0.01,0.1,1,10,100] and γ=['scale','auto',0.001,0.1]
══════════════════════════════════════════════════════════
```
