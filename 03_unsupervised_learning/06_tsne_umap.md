# t-SNE and UMAP — Non-Linear Dimensionality Reduction

## 1. Overview

Both t-SNE and UMAP are **non-linear** dimensionality reduction methods primarily used for **visualisation** of high-dimensional data. They preserve local neighbourhood structure.

| | t-SNE | UMAP | PCA |
|--|-------|------|-----|
| Type | Non-linear | Non-linear | Linear |
| Speed | Slow O(n log n) | Fast O(n^{1.14}) | Fast O(nd²) |
| Preserves | Local structure | Local + some global | Global variance |
| Global structure | ❌ Poor | ✅ Better | ✅ Good |
| Reproducible | ❌ (random init) | ✅ (with random_state) |✅ |
| New points | ❌ No transform | ✅ Has transform | ✅ Has transform |
| Interpretable axes | ❌ No | ❌ No | ✅ (loadings) |

---

## 2. t-SNE — How It Works

### 2.1 High-Dimensional Similarities

For each pair of points (i, j) in the original d-dimensional space, compute a conditional probability that i would pick j as a neighbour:

```
p_{j|i} = exp(−‖xᵢ−xⱼ‖² / 2σᵢ²) / Σ_{k≠i} exp(−‖xᵢ−xₖ‖² / 2σᵢ²)

p_{ij} = (p_{j|i} + p_{i|j}) / 2n    (symmetrised)
```

σᵢ is chosen such that Perp(Pᵢ) = 2^{H(Pᵢ)} equals the specified `perplexity`, where H(Pᵢ) is the Shannon entropy of the distribution Pᵢ. This is found via binary search on σᵢ.

**Perplexity ≈ effective number of neighbours considered.**  
Typical range: 5–50. Common default: 30.

### 2.2 Low-Dimensional Similarities (Student t-distribution)

In the 2D embedding space, similarities are modelled with a **Student t-distribution with 1 degree of freedom** (Cauchy distribution):

```
q_{ij} = (1 + ‖yᵢ−yⱼ‖²)⁻¹ / Σ_{k≠l} (1 + ‖yₖ−yₗ‖²)⁻¹
```

**Why t-distribution instead of Gaussian?** The Gaussian in low dimensions creates a "crowding problem" — moderate distances in high-d cannot be faithfully represented in 2D with a Gaussian because the volume of a sphere grows with dimension. The heavy tail of the t-distribution pushes moderately-distant points further apart, alleviating crowding.

### 2.3 KL Divergence Objective

t-SNE minimises the **KL divergence** between the high-d and low-d similarity distributions:

```
KL(P ‖ Q) = Σᵢⱼ p_{ij} · log(p_{ij} / q_{ij})
```

Minimised via **gradient descent**. The gradient for each embedding point yᵢ has an attractive force (for close pairs in high-d) and a repulsive force (for all other pairs):

```
∂KL/∂yᵢ = 4 Σⱼ (p_{ij} − q_{ij})(yᵢ − yⱼ)(1 + ‖yᵢ−yⱼ‖²)⁻¹
```

---

## 3. UMAP — How It Works

UMAP (Uniform Manifold Approximation and Projection) is grounded in **Riemannian geometry and algebraic topology**.

### High-Level Steps:

```
1. Build a weighted k-nearest-neighbour graph in the high-dimensional space.
   - Weights encode fuzzy set membership based on distance to each point's
     k nearest neighbours, normalised by the distance to the nearest neighbour
     (making the metric locally adaptive).

2. Construct a low-dimensional representation that is "topologically equivalent"
   — i.e., has the same fuzzy topological structure (simplicial complex).

3. Minimise cross-entropy between high-d and low-d fuzzy simplicial sets:
   CE = Σᵢⱼ [ w_{ij}·log(w_{ij}/v_{ij}) + (1−w_{ij})·log((1−w_{ij})/(1−v_{ij})) ]

   where w_{ij} are high-d weights and v_{ij} are low-d weights.

4. Use stochastic gradient descent with negative sampling for speed.
```

**Key parameters:**
- `n_neighbors` (5–100): controls local vs. global structure preservation. Small → local, large → global.
- `min_dist` (0.0–1.0): controls tightness of clusters in the embedding. Small → tighter clusters.
- `metric`: distance metric for the high-d graph (euclidean, cosine, etc.)

---

## 4. Perplexity Effect — Visualisation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

perplexities = [5, 20, 50, 100]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, perp in zip(axes, perplexities):
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter=1000,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    X_2d = tsne.fit_transform(X_scaled)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    ax.set_title(f'perplexity={perp}\nKL={tsne.kl_divergence_:.2f}')
    ax.axis('off')

plt.suptitle('t-SNE: Effect of Perplexity on Digits Dataset', fontsize=13)
plt.tight_layout()
plt.savefig('tsne_perplexity_effect.png', dpi=100)
plt.show()
```

**Perplexity guidelines:**
- **Too low (< 5):** Tiny, fragmented clusters — each point only looks at a few neighbours.
- **Optimal (15–50):** Well-separated, meaningful clusters.
- **Too high (> n/5):** Clusters merge — each point considers too many neighbours.
- Rule of thumb: perplexity ∈ [max(5, √n), min(50, n/5)].

---

## 5. t-SNE vs UMAP vs PCA — Decision Guide

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# Try to import UMAP (optional install)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Install UMAP: pip install umap-learn")

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# PCA (fast, linear)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# t-SNE (slower, non-linear)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
            learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

embeddings = {'PCA': X_pca, 't-SNE': X_tsne}

if HAS_UMAP:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    embeddings['UMAP'] = X_umap

fig, axes = plt.subplots(1, len(embeddings), figsize=(6 * len(embeddings), 5))
if len(embeddings) == 1:
    axes = [axes]

for ax, (name, X_2d) in zip(axes, embeddings.items()):
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    ax.set_title(name)
    ax.axis('off')

plt.colorbar(sc, ax=axes[-1], label='Digit')
plt.suptitle('Comparison: PCA vs t-SNE vs UMAP (Digits)', fontsize=13)
plt.tight_layout()
plt.savefig('tsne_umap_pca_comparison.png', dpi=100)
plt.show()
```

---

## 6. When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Quick exploration, interpretable axes | **PCA** |
| Large dataset (n > 100 000) | **UMAP** (fast) or PCA |
| Publication-quality 2D visualisation | **t-SNE** (most visually appealing) |
| Preserve global structure | **PCA** or **UMAP** (large `n_neighbors`) |
| Need to project new points | **PCA** or **UMAP** (`.transform()`) |
| Downstream ML features | **PCA** (interpretable, stable) |
| Biological data (scRNA-seq) | **UMAP** (fast, preserves trajectories) |
| NLP embedding visualisation | **UMAP** or **t-SNE** |

---

## 7. Critical Limitations of t-SNE

1. **Distances between clusters are meaningless**  
   KL divergence is asymmetric; the relative distances between cluster centres in the 2D plot do NOT reflect true distances in high-d. Two clusters far apart in t-SNE may be close in the original space.

2. **Cluster sizes are misleading**  
   t-SNE normalises point densities, so all clusters appear roughly the same size regardless of true density.

3. **No inverse transform / out-of-sample**  
   sklearn's TSNE has no `.transform()` method — you cannot project new points. Workaround: fit PCA first, then t-SNE on PCA output.

4. **Non-deterministic**  
   Results vary between runs unless you set `random_state`. Even then, the arrangement can flip or rotate.

5. **Slow for large n**  
   O(n log n) with Barnes-Hut approximation (default), but still slow for n > 50 000. Use PCA to reduce to ~50 dims first, then t-SNE.

6. **Perplexity sensitivity**  
   Results can look completely different with different perplexity values; always try multiple values and compare.

7. **Multiple runs required for trust**  
   Run t-SNE 3+ times; if the same cluster structure appears consistently, it is likely real.

---

## 8. Best Practices

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

# Step 1: Standardise
X_scaled = StandardScaler().fit_transform(X)

# Step 2: PCA pre-reduction (speeds up t-SNE, removes noise)
pca_50 = PCA(n_components=50, random_state=42)
X_pca50 = pca_50.fit_transform(X_scaled)
print(f"PCA variance retained: {pca_50.explained_variance_ratio_.sum():.3f}")

# Step 3: t-SNE with best practices
tsne = TSNE(
    n_components=2,
    perplexity=30,           # try 5, 30, 50
    n_iter=1000,             # ≥ 1000 for convergence
    learning_rate='auto',    # sklearn ≥ 1.2: 'auto' ≈ max(200, n/12)
    init='pca',              # PCA init reduces randomness
    random_state=42,
    n_jobs=-1                # parallelise
)
X_2d = tsne.fit_transform(X_pca50)
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
# Lower KL divergence = better fit; compare across perplexity values
```

---

## 9. Interview Q&A

**Q1: Why does t-SNE use a t-distribution in the low-dimensional space?**  
A: To address the **crowding problem**. In high dimensions, points at moderate distances occupy a large volume. When projecting to 2D, there isn't enough space to represent all these moderate distances — points get crowded. The t-distribution's heavier tail pushes moderately-distant points further apart in 2D, creating clearer separation between clusters.

**Q2: What is perplexity in t-SNE?**  
A: Perplexity can be interpreted as the effective number of nearest neighbours considered for each point. Formally, it is 2^{entropy of the conditional distribution Pᵢ}. It controls the bandwidth σᵢ of the Gaussian kernel in high-d similarity. Lower perplexity = finer local structure; higher perplexity = more global view. Typical range: 5–50.

**Q3: Can you use t-SNE embeddings as features for a classifier?**  
A: Generally **no**. t-SNE is a visualisation tool — the axes are not interpretable, distances between clusters are meaningless, and there is no `.transform()` for new data. For feature engineering, use PCA, autoencoders, or UMAP (which has `.transform()`).

**Q4: How is UMAP faster than t-SNE?**  
A: UMAP uses approximate nearest neighbours (with a forest of random projection trees), then uses stochastic gradient descent with negative sampling to optimise the embedding. This gives roughly O(n^{1.14}) complexity vs. t-SNE's O(n log n) with Barnes-Hut, and UMAP's constant factor is much smaller in practice.

**Q5: What is the key difference between t-SNE and UMAP regarding global structure?**  
A: t-SNE is almost entirely local — it matches nearest-neighbour similarities and pays little attention to the relative positions of distant clusters. UMAP preserves more global structure because its graph construction is based on a uniform manifold assumption; the relative distances between clusters in UMAP plots tend to be more meaningful than in t-SNE.

**Q6: What does a low KL divergence mean in t-SNE?**  
A: A lower KL divergence means the 2D embedding more faithfully represents the high-d similarity structure. However, comparing KL divergence across different perplexity values is not valid — KL divergence depends on the high-d probabilities, which change with perplexity.

---

## 10. Common Pitfalls

1. **Interpreting inter-cluster distances in t-SNE** — meaningless; do not say "cluster A is far from cluster B so they are very different."
2. **Interpreting cluster sizes in t-SNE** — all clusters appear normalised; a big visual cluster is not necessarily a large or dense cluster.
3. **Running t-SNE once** — results can vary; run multiple times and look for consistent patterns.
4. **Skipping PCA pre-reduction** — t-SNE on raw 1000-d data is very slow; reduce to 50 dims with PCA first.
5. **Using t-SNE output as ML features** — no transform for new points; use PCA or UMAP instead.
6. **Treating different perplexity plots as the same** — always label the perplexity used; different perplexities give different layouts.
7. **Not standardising before t-SNE/UMAP** — same issue as K-Means; large-variance features dominate distances.

---

## 11. Quick Reference Cheat Sheet

```
t-SNE:
  Objective:   Minimise KL(P‖Q); P=high-d Gaussian, Q=low-d t-distribution
  Perplexity:  Effective neighbours; try 5, 20, 30, 50
  Init:        Use init='pca' for reproducibility and faster convergence
  Pre-reduce:  PCA to 50 dims first (speed + noise reduction)
  Limitations: No transform, inter-cluster distances meaningless, slow
  Best for:    Single-run visualisation of local cluster structure

UMAP:
  Objective:   Match fuzzy topological structure (cross-entropy)
  n_neighbors: Controls local vs global (5=local, 100=global)
  min_dist:    Cluster compactness (0.0=tight, 1.0=spread)
  Advantages:  Fast, has .transform(), preserves more global structure
  Best for:    Large datasets, when new-point projection needed

PCA:
  Best for:    Pre-processing, interpretable components, fast exploration

Decision:
  Visualisation only      → t-SNE (most visually appealing)
  Large dataset / speed   → UMAP
  Pre-processing for ML   → PCA
  New-point projection    → PCA or UMAP

sklearn t-SNE:
  from sklearn.manifold import TSNE
  tsne = TSNE(n_components=2, perplexity=30, init='pca',
              learning_rate='auto', random_state=42)
  X_2d = tsne.fit_transform(X_scaled)
```
