# Principal Component Analysis (PCA)

## 1. Core Concept

PCA finds the directions (**principal components**) of **maximum variance** in high-dimensional data and projects the data onto a lower-dimensional subspace defined by the top k components. It is the most widely used linear dimensionality reduction technique.

**Goals:**
- Remove redundancy (correlated features).
- Reduce noise (discard low-variance directions).
- Visualise high-dimensional data in 2D/3D.
- Speed up downstream ML by reducing features.

---

## 2. Eigendecomposition of the Covariance Matrix

**Step-by-step derivation:**

```
Given: data matrix X ∈ ℝⁿˣᵈ (n samples, d features), mean-centred.

1. Compute covariance matrix:
   C = (1/(n-1)) · Xᵀ · X   ∈ ℝᵈˣᵈ

2. Eigendecompose C:
   C = V · Λ · Vᵀ

   where:
     V = [v₁ | v₂ | … | vᵈ]  (eigenvectors = principal directions)
     Λ = diag(λ₁, λ₂, …, λᵈ)  (eigenvalues, λ₁ ≥ λ₂ ≥ … ≥ λᵈ ≥ 0)

3. Project data onto top-k eigenvectors:
   Z = X · Vₖ   ∈ ℝⁿˣᵏ   (scores / embeddings)

4. Reconstruct (lossy):
   X̂ = Z · Vₖᵀ
```

**Why eigenvectors?** The first eigenvector v₁ is the direction that maximises Var(Xv₁) = v₁ᵀCv₁ subject to ‖v₁‖=1. By the Rayleigh quotient, the solution is the largest eigenvector. Subsequent components are orthogonal to all previous ones and capture the next most variance.

**Variance captured by component k = λₖ / Σᵢ λᵢ**

---

## 3. Explained Variance Ratio

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load real dataset: 30 features
data = load_breast_cancer()
X = data.data
y = data.target

# Always standardise before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA with all components
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

evr = pca_full.explained_variance_ratio_
cumulative_evr = np.cumsum(evr)

# Number of components for 95% variance
n_95 = np.argmax(cumulative_evr >= 0.95) + 1
print(f"Components for 95% variance: {n_95} (out of {X.shape[1]})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Individual explained variance (Scree plot)
axes[0].bar(range(1, len(evr) + 1), evr, alpha=0.7, label='Individual EVR')
axes[0].plot(range(1, len(evr) + 1), evr, 'ro-', markersize=5)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot')
axes[0].axvline(x=n_95, color='blue', linestyle='--')

# Cumulative explained variance
axes[1].plot(range(1, len(cumulative_evr) + 1), cumulative_evr, 'b-o', markersize=5)
axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
axes[1].axvline(x=n_95, color='green', linestyle='--', label=f'{n_95} components')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()

plt.tight_layout()
plt.savefig('pca_explained_variance.png', dpi=100)
plt.show()
```

---

## 4. Scree Plot for Choosing Components

The **scree plot** (eigenvalue vs. component number) helps identify the "elbow" — the point after which eigenvalues level off, suggesting the remaining components capture mostly noise.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# 64-feature digit dataset
X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(random_state=42)
pca.fit(X_scaled)

eigenvalues = pca.explained_variance_  # unnormalised eigenvalues

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'ko-', markersize=5)
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot — Eigenvalues')
plt.axvline(x=10, color='red', linestyle='--', label='Elbow ≈ 10')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('pca_scree.png', dpi=100)
plt.show()

# Practical rule: keep components with eigenvalue ≥ 1 (Kaiser criterion)
kaiser_n = (eigenvalues >= 1).sum()
print(f"Kaiser criterion: keep {kaiser_n} components (eigenvalue ≥ 1)")
```

**Rules of thumb for choosing k:**
1. **Elbow method:** Visual kink in the scree plot.
2. **95% cumulative explained variance:** Most common threshold.
3. **Kaiser criterion:** Keep components with eigenvalue ≥ 1 (only for standardised data).
4. **Task-driven:** Pick the smallest k that gives acceptable downstream model performance.

---

## 5. PCA for Visualisation, Clustering, and Neural Networks

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# --- 1. Visualisation: project to 2D ---
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('PCA 2D Projection of Digits Dataset')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)')
plt.tight_layout()
plt.savefig('pca_digits_2d.png', dpi=100)
plt.show()

# --- 2. Before Clustering: does PCA help K-Means? ---
results = {}
for k_pca in [2, 5, 10, 20, 64]:
    if k_pca == 64:
        X_in = X_scaled
    else:
        pca_k = PCA(n_components=k_pca, random_state=42)
        X_in = pca_k.fit_transform(X_scaled)
    km = KMeans(n_clusters=10, init='k-means++', n_init=5, random_state=42)
    labels = km.fit_predict(X_in)
    sil = silhouette_score(X_in, labels)
    results[k_pca if k_pca != 64 else 'raw'] = round(sil, 3)

print("Silhouette score by PCA components:")
for k, v in results.items():
    print(f"  {k:>4} components: {v}")

# --- 3. Before Neural Network: noise reduction ---
pca_50 = PCA(n_components=0.95, random_state=42)  # auto-select for 95% variance
X_compressed = pca_50.fit_transform(X_scaled)
print(f"\nPCA keeps {pca_50.n_components_} components (95% variance)")
print(f"Compression ratio: {X.shape[1]} → {pca_50.n_components_} "
      f"({X.shape[1] / pca_50.n_components_:.1f}x)")
```

---

## 6. PCA vs SVD Connection

PCA via covariance eigendecomposition and PCA via SVD are mathematically equivalent. SVD is numerically superior.

```
Covariance-based:
  C = XᵀX/(n-1) = VΛVᵀ
  Principal components: columns of V
  Scores: XV

SVD-based:
  X = UΣVᵀ   (thin/economy SVD, U∈ℝⁿˣᵏ, Σ∈ℝᵏˣᵏ, V∈ℝᵈˣᵏ)
  Relation: eigenvalues λᵢ = σᵢ²/(n-1)
  Scores: XV = UΣ
  sklearn.decomposition.PCA internally uses SVD for numerical stability.
```

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

X, _ = load_breast_cancer(return_X_y=True)
X_c = StandardScaler().fit_transform(X)

# SVD approach
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
scores_svd = X_c @ Vt[:2].T  # project onto top-2 right singular vectors

# Covariance approach
cov = X_c.T @ X_c / (len(X_c) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
scores_eig = X_c @ eigenvectors[:, :2]

# They should match (up to sign flip)
print("Max abs diff:", np.max(np.abs(np.abs(scores_svd) - np.abs(scores_eig))))
```

**TruncatedSVD** — PCA without centering (important for sparse matrices, e.g., TF-IDF):

```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

# Sparse TF-IDF matrix
X_sparse = sparse_random(1000, 5000, density=0.02, random_state=42)
svd = TruncatedSVD(n_components=50, random_state=42)
X_lsa = svd.fit_transform(X_sparse)  # Latent Semantic Analysis
print(f"Compressed: {X_sparse.shape} → {X_lsa.shape}")
```

---

## 7. Incremental PCA (Large Datasets)

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

np.random.seed(42)
X_large = np.random.randn(100_000, 50)

ipca = IncrementalPCA(n_components=10, batch_size=1000)
X_reduced = ipca.fit_transform(X_large)
print(f"IncrementalPCA: {X_large.shape} → {X_reduced.shape}")
print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.3f}")
```

---

## 8. Interview Q&A

**Q1: What does PCA actually do, intuitively?**  
A: PCA rotates the coordinate system to align axes with the directions of maximum variance in the data. The first axis (PC1) points in the direction of greatest spread, PC2 is orthogonal to PC1 and captures the next most variance, and so on. It's like finding the "natural axes" of an ellipsoid fitted to the data. This removes correlation between features and orders directions by importance.

**Q2: Why must you standardise data before PCA?**  
A: PCA finds directions of maximum *variance*. A feature measured in thousands (e.g., salary) will dominate a feature measured in single digits (e.g., age) purely because of scale, not because it is more important. StandardScaler ensures all features contribute equally on a variance basis.

**Q3: What is the difference between PCA and SVD?**  
A: PCA via covariance eigendecomposition and PCA via SVD are mathematically equivalent. SVD is numerically more stable (avoids squaring the data matrix) and works on the data matrix directly. sklearn's PCA uses SVD internally. TruncatedSVD is a variant that skips mean-centering and works on sparse matrices (useful for NLP/TF-IDF).

**Q4: How do you choose the number of components?**  
A: (1) Scree plot elbow, (2) 95% cumulative explained variance threshold, (3) Kaiser criterion (eigenvalue ≥ 1), (4) Downstream task performance. Set `n_components=0.95` in sklearn to automatically select the minimum number of components explaining 95% of variance.

**Q5: What are the limitations of PCA?**  
A: (1) Linear only — cannot capture nonlinear structure (→ Kernel PCA, t-SNE, UMAP). (2) Maximises variance, not class separability — for supervised tasks, use LDA. (3) Components are hard to interpret — they are linear combinations of all features. (4) Sensitive to outliers — one extreme point can shift principal directions significantly (→ Robust PCA).

**Q6: Can PCA be used for noise reduction?**  
A: Yes. Low-eigenvalue components correspond to noise directions. By discarding them and reconstructing, we get a denoised version: X̂ = Z · Vₖᵀ. This works well for images and signal processing.

**Q7: When should you NOT use PCA?**  
A: (1) When features are already uncorrelated (PCA adds no benefit). (2) When interpretability of individual features matters (components are mixtures). (3) When the relationship between features and target is nonlinear and you need to preserve it. (4) When data is sparse (use TruncatedSVD instead).

---

## 9. Common Pitfalls

1. **Not scaling before PCA** — high-variance features dominate; always `StandardScaler` first.
2. **Applying PCA to test data with different fit** — always `fit` on training data, `transform` on test.
3. **Choosing k by explained variance alone** — 95% may not be the right threshold for your task; validate downstream.
4. **Using PCA on sparse matrices** — centering destroys sparsity; use `TruncatedSVD` instead.
5. **Interpreting PC directions as causal** — PCs are mathematical constructs, not causal variables.
6. **Forgetting that PCA is lossy** — reconstruction error exists; quantify it for critical applications.
7. **Applying PCA before classification without re-evaluating** — sometimes PCA discards discriminative but low-variance directions; use LDA or supervised feature selection instead.

---

## 10. Quick Reference Cheat Sheet

```
Goal:        Reduce dimensions by projecting onto directions of max variance
Math:        Eigendecompose covariance C = VΛVᵀ; project X → XVₖ
Scaling:     MANDATORY — StandardScaler before PCA
Choose k:    Scree plot elbow | 95% cumulative EVR | Kaiser (λ≥1)
n_components=0.95 in sklearn → auto-select for 95% variance
SVD link:    X = UΣVᵀ; eigenvalues = σᵢ²/(n-1)
Sparse data: Use TruncatedSVD (no centering)
Large n:     Use IncrementalPCA (batch processing)
Limitation:  Linear only; hard to interpret; sensitive to outliers
Fit/transform: Fit on TRAIN only; transform train and test separately

sklearn quick start:
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler
  X_s   = StandardScaler().fit_transform(X_train)
  pca   = PCA(n_components=0.95, random_state=42)
  X_pca = pca.fit_transform(X_s)       # train
  X_test_pca = pca.transform(X_test_s) # test
  print(pca.explained_variance_ratio_)
  print(pca.n_components_)
```
