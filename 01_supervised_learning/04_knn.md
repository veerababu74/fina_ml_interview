# 🔍 K-Nearest Neighbors — Complete Interview Guide

> **5-Pillar Answer Framework:** WHAT | WHY | WHEN | HOW | PITFALLS

---

## 🏛️ The 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | A lazy, non-parametric algorithm that classifies/predicts by majority vote/average of the K nearest training examples in feature space |
| **WHY** | No training time, naturally handles multi-class, works for both classification and regression, simple to understand |
| **WHEN** | Small-to-medium datasets, low-dimensional spaces, simple baseline, recommendation systems, anomaly detection |
| **HOW** | For each query point, compute distances to all training points, find K nearest, return majority class (or average for regression) |
| **PITFALLS** | Slow inference O(n·d) per query, fails with high dimensions (curse of dimensionality), sensitive to scale and irrelevant features |

---

## 📐 Mathematical Foundation

### Prediction Rules

**KNN Classification:**
```
ŷ = argmax_c Σᵢ∈Nₖ(x) I(yᵢ = c)
```
Majority vote among K nearest neighbors.

**KNN Regression:**
```
ŷ = 1/K Σᵢ∈Nₖ(x) yᵢ
```
Average value of K nearest neighbors.

**Weighted KNN:**
```
ŷ = Σᵢ∈Nₖ(x) wᵢ·yᵢ / Σᵢ wᵢ    where wᵢ = 1/d(x, xᵢ)
```
Closer neighbors have more influence.

---

## 📏 Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean (L2)** | √Σ(xᵢ-zᵢ)² | Continuous, dense features, similar scales |
| **Manhattan (L1)** | Σ\|xᵢ-zᵢ\| | Sparse data, high-dim, more robust to outliers |
| **Minkowski** | (Σ\|xᵢ-zᵢ\|^p)^(1/p) | Generalizes L1 (p=1) and L2 (p=2) |
| **Chebyshev (L∞)** | max\|xᵢ-zᵢ\| | When max difference matters |
| **Hamming** | Fraction of positions differing | Binary/categorical features |
| **Cosine** | 1 - (x·z)/(‖x‖‖z‖) | Text data, directional similarity |
| **Mahalanobis** | √(x-z)ᵀΣ⁻¹(x-z) | Correlated features, scale-invariant |

---

## 📊 Choosing K

| K Value | Behavior | Bias | Variance |
|---------|----------|------|----------|
| K=1 | Voronoi diagram, overfit | Very Low | Very High |
| Small K | Complex, noisy boundary | Low | High |
| Optimal K | Smooth but accurate | Medium | Medium |
| Large K | Smooth, underfit | High | Low |
| K=n | Predict majority class always | Very High | Very Low |

**Rules of thumb:**
- Start with K = √n (square root of training samples)
- Always use **odd K** for binary classification (avoid ties)
- Select via cross-validation on a range [1, 2, ..., 30]
- Larger K for noisy data

---

## 💀 Curse of Dimensionality

As dimensions d increase:
- Volume of unit hypercube = 1
- Volume of unit ball = π^(d/2) / Γ(d/2+1) → **0 as d → ∞**
- All points become equidistant → distance metrics lose meaning
- Fraction of data in nearest neighborhood shrinks exponentially

**Practical impact:**
```
# In 1D, 10% of data within 0.1 radius
# In 10D, need 63% radius to capture 10% of data
# In 100D, need 99.9% radius!
fraction = k/n
required_radius = fraction^(1/d)
```

**Solutions:** PCA/UMAP before KNN, feature selection, use Mahalanobis distance, use tree-based models instead.

---

## ⚡ KD-Tree and Ball-Tree for Speed

| Structure | Build Time | Query Time | Best For |
|-----------|-----------|-----------|---------|
| **Brute Force** | O(1) | O(n·d) | Any, small n |
| **KD-Tree** | O(n·log n) | O(log n) avg | Low-dim (d < 20), Euclidean |
| **Ball-Tree** | O(n·log n) | O(log n) avg | High-dim, non-Euclidean metrics |

**KD-Tree:** Recursively partitions space along axis with highest variance. Query: prune branches too far from query point.

**Ball-Tree:** Partitions into nested hyperspheres. More efficient than KD-Tree for d > 20.

---

## 💻 Full Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, train_test_split
)
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification, load_iris
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ── Dataset ───────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=6,
    n_classes=3, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 1. Basic KNN Classifier ───────────────────────────────────────
# ALWAYS scale features for KNN — distances are scale-sensitive!
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',    # 'uniform' or 'distance' (inverse distance weighting)
        metric='minkowski',   # 'euclidean', 'manhattan', 'minkowski', 'cosine'
        p=2,                  # p=2 → Euclidean, p=1 → Manhattan
        algorithm='auto',     # 'auto', 'ball_tree', 'kd_tree', 'brute'
        leaf_size=30,         # For ball_tree/kd_tree
        n_jobs=-1             # Parallel computation
    ))
])
knn_pipeline.fit(X_train, y_train)
print(f"KNN (k=5) Accuracy: {knn_pipeline.score(X_test, y_test):.4f}")

# ── 2. Choosing Optimal K via Cross-Validation ────────────────────
k_range = range(1, 31)
k_scores = []

for k in k_range:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=k, weights='uniform'))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"\nOptimal K: {optimal_k} (CV Accuracy: {max(k_scores):.4f})")

# Plot K vs Accuracy
plt.figure(figsize=(10, 5))
plt.plot(k_range, k_scores, 'bo-', linewidth=2, markersize=6)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.xlabel('K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Choosing Optimal K')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('knn_k_selection.png', dpi=100)
print("K-selection plot saved.")

# ── 3. GridSearchCV: K + metric + weights ────────────────────────
param_grid = {
    'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan', 'minkowski'],
    'model__p': [1, 2]
}

grid_search = GridSearchCV(
    estimator=Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier())]),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest KNN params: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Test Accuracy: {grid_search.score(X_test, y_test):.4f}")

# ── 4. KNN Regression ─────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

X_r, y_r = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42
)

knn_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',   # Weighted average — better than uniform for regression
        algorithm='ball_tree'
    ))
])
knn_reg.fit(X_r_train, y_r_train)
y_r_pred = knn_reg.predict(X_r_test)
rmse = np.sqrt(mean_squared_error(y_r_test, y_r_pred))
r2 = knn_reg.score(X_r_test, y_r_test)
print(f"\nKNN Regression — RMSE: {rmse:.2f}, R²: {r2:.4f}")

# ── 5. KNN from Scratch ───────────────────────────────────────────
class KNNFromScratch:
    """K-Nearest Neighbors implemented from scratch."""

    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric

    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_nearest_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_idx]
        # Majority vote
        counts = np.bincount(k_nearest_labels.astype(int))
        return np.argmax(counts)


# Test from-scratch implementation
from sklearn.preprocessing import LabelEncoder
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_i_train_s = scaler.fit_transform(X_i_train)
X_i_test_s = scaler.transform(X_i_test)

knn_scratch = KNNFromScratch(k=5)
knn_scratch.fit(X_i_train_s, y_i_train)
y_scratch_pred = knn_scratch.predict(X_i_test_s)
accuracy = np.mean(y_scratch_pred == y_i_test)
print(f"\nKNN from scratch Iris accuracy: {accuracy:.4f}")
```

---

## ❓ Interview Questions & Answers

### Q1: What is the curse of dimensionality and how does it affect KNN?
**Answer:** In high dimensions, all pairwise distances converge to the same value — points become equidistant. For KNN this is catastrophic: the K nearest neighbors are no more similar than random points. Additionally, to maintain a fixed fraction of training data within a neighborhood, the radius must grow to near the data range. Solutions: PCA/feature selection before KNN, switch to tree models for high-dim data.

### Q2: How do you choose K?
**Answer:** Use **cross-validation**: train with K ∈ [1, 30], pick K maximizing validation accuracy. General rules: (1) K = √n as starting point, (2) Larger K for noisier data, (3) Always odd K for binary classification to avoid ties, (4) K=1 memorizes data (perfect train accuracy), (5) Large K gives smoother boundary. Plot validation accuracy vs K — look for the "elbow."

### Q3: Why is feature scaling critical for KNN?
**Answer:** KNN computes distances like Euclidean: d = √Σ(xᵢ-zᵢ)². A feature with range [0, 1000] will dominate one with range [0, 1]. Without scaling, KNN effectively uses only the high-scale features. StandardScaler (z-score) or MinMaxScaler must be applied before KNN. This is the single most important preprocessing step for KNN.

### Q4: KNN is a "lazy learner" — what does that mean?
**Answer:** Lazy learning = no explicit training phase. KNN memorizes all training data and defers computation to prediction time. Eager learners (logistic regression, SVM) build a model during training and discard training data. KNN's training is O(1) but prediction is O(n·d) — the opposite of eager learners. Memory usage is O(n·d) — all data must be stored.

### Q5: When would you prefer weighted KNN over uniform KNN?
**Answer:** Use **weighted KNN** (weights='distance') when: training data is noisy (closer neighbors should matter more), data has varying density, K is large. Weights = 1/distance give closer points more influence, reducing sensitivity to the choice of K. For regression tasks, weighted KNN is usually better. Note: if there's a tie with uniform, 'distance' weighting breaks it naturally.

---

## ⚠️ Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Not scaling features | Dominated by high-range features | StandardScaler or MinMaxScaler |
| High-dimensional data | All distances similar, poor accuracy | PCA, feature selection first |
| Large n | Slow prediction O(n·d) | Ball-tree, approximate NN (FAISS, Annoy) |
| K=1 | Perfect train accuracy, poor test | Use CV to select K |
| Even K with binary classes | Ties possible | Use odd K |
| Memory issues | Can't fit data in RAM | Mini-batch approximations, LSH |

---

## 📋 Quick Reference Cheat Sheet

```
KNN Quick Reference
══════════════════════════════════════════════════════════
Classification: ŷ = majority_vote(K nearest neighbors)
Regression:     ŷ = mean(K nearest neighbors)
Weighted:       ŷ = Σ(wᵢ·yᵢ)/Σwᵢ  where wᵢ = 1/dist(x, xᵢ)

Distance: Euclidean (default), Manhattan, Cosine (text)
Choosing K: cross-validate [1..30], start with K=√n, use odd K
Complexity: Train O(1), Predict O(n·d)
Speed up: KD-Tree (d<20), Ball-Tree (d>20), FAISS for millions

MUST scale features! (StandardScaler)
Algorithm param: 'auto' chooses kd_tree/ball_tree/brute automatically

sklearn:
  KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
  KNeighborsRegressor(n_neighbors=7, weights='distance')
══════════════════════════════════════════════════════════
```
