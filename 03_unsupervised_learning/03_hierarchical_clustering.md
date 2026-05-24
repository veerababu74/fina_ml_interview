# Hierarchical Clustering

## 1. Core Concept

Hierarchical clustering builds a **tree of clusters (dendrogram)** by successively merging (agglomerative) or splitting (divisive) clusters. Unlike K-Means, it does not require k in advance and reveals multi-scale cluster structure.

Two main strategies:
- **Agglomerative (bottom-up):** Start with n singleton clusters; merge the two closest at each step until one cluster remains. ← Most common.
- **Divisive (top-down):** Start with one cluster containing all points; recursively split. Rarely used in practice (O(2ⁿ) exhaustive search).

---

## 2. Agglomerative Algorithm — Step by Step

```
Input:  n data points, distance metric d(·,·), linkage criterion L
Output: dendrogram of merges

1. Initialise n clusters: C₁={x₁}, C₂={x₂}, …, Cₙ={xₙ}
2. Compute n×n pairwise distance matrix D.
3. REPEAT until only 1 cluster remains:
   a. Find the two clusters Cᵢ, Cⱼ with minimum L(Cᵢ, Cⱼ).
   b. Record merge (Cᵢ, Cⱼ, merge_height).
   c. Replace Cᵢ and Cⱼ with Cᵢ ∪ Cⱼ.
   d. Update D using the Lance–Williams formula.
4. Return the full merge history as a dendrogram.
```

**Time complexity:** O(n³) naive; O(n² log n) with priority queues.  
**Space complexity:** O(n²) for the distance matrix.

**Lance–Williams formula** (general linkage update):
```
d(Cₖ, Cᵢ∪Cⱼ) = αᵢ·d(Cₖ,Cᵢ) + αⱼ·d(Cₖ,Cⱼ) + β·d(Cᵢ,Cⱼ) + γ·|d(Cₖ,Cᵢ) − d(Cₖ,Cⱼ)|
```
Different linkage criteria correspond to different (αᵢ, αⱼ, β, γ) values.

---

## 3. Linkage Methods

### 3.1 Single Linkage (Minimum)
```
d(Cᵢ, Cⱼ) = min_{x∈Cᵢ, y∈Cⱼ} d(x, y)
```
- Uses the **closest pair** of points between clusters.
- Prone to **chaining**: elongated, chain-like clusters.
- Good for detecting non-convex shapes.
- (αᵢ=½, αⱼ=½, β=0, γ=−½)

### 3.2 Complete Linkage (Maximum)
```
d(Cᵢ, Cⱼ) = max_{x∈Cᵢ, y∈Cⱼ} d(x, y)
```
- Uses the **farthest pair** — ensures all points in merged cluster are within diameter.
- Produces compact, roughly equal-sized clusters.
- Sensitive to outliers.
- (αᵢ=½, αⱼ=½, β=0, γ=+½)

### 3.3 Average Linkage (UPGMA)
```
d(Cᵢ, Cⱼ) = (1/|Cᵢ||Cⱼ|) Σ_{x∈Cᵢ} Σ_{y∈Cⱼ} d(x, y)
```
- Average pairwise distance — compromise between single and complete.
- Less sensitive to outliers than complete linkage.
- (αᵢ=|Cᵢ|/(|Cᵢ|+|Cⱼ|), αⱼ=|Cⱼ|/(|Cᵢ|+|Cⱼ|), β=0, γ=0)

### 3.4 Ward's Linkage (Minimum Variance)
```
d(Cᵢ, Cⱼ) = Δ WCSS = |Cᵢ||Cⱼ|/(|Cᵢ|+|Cⱼ|) · ‖μᵢ − μⱼ‖²
```
- Merges the pair of clusters that minimises the **increase in total WCSS**.
- Produces compact, similarly-sized clusters.
- **Most commonly used for numerical data** — often gives the best results.
- Requires Euclidean distance.

| Linkage | Sensitive to Outliers | Cluster Shape | Best For |
|---------|----------------------|---------------|---------|
| Single | Low | Elongated, chains | Non-convex, detecting bridges |
| Complete | High | Compact, spherical | Well-separated, equal-size |
| Average | Medium | Moderate | General purpose |
| Ward | Low | Compact, equal-size | Numerical data, general purpose |

---

## 4. Full Example with Dendrogram

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate data
X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=0.6, random_state=42)
X = StandardScaler().fit_transform(X)

# Compare all linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, method in zip(axes, linkage_methods):
    Z = linkage(X, method=method)
    dendrogram(
        Z,
        ax=ax,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True
    )
    ax.set_title(f'{method.capitalize()} Linkage')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')

plt.suptitle('Dendrogram Comparison Across Linkage Methods', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('hierarchical_dendrograms.png', dpi=100)
plt.show()

# Cut Ward dendrogram at k=4 clusters
Z_ward = linkage(X, method='ward')
labels_4 = fcluster(Z_ward, t=4, criterion='maxclust')
sil = silhouette_score(X, labels_4)
print(f"Ward k=4 Silhouette Score: {sil:.3f}")
```

---

## 5. Reading and Cutting a Dendrogram

```
Height (y-axis): distance at which two clusters merged.

How to read:
  - Long vertical lines → big jump in merge distance → natural cluster boundary.
  - Cut horizontally across the longest vertical lines to get k clusters.

How to cut programmatically:
  - By number of clusters: fcluster(Z, t=k, criterion='maxclust')
  - By distance threshold: fcluster(Z, t=d, criterion='distance')
```

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
X = StandardScaler().fit_transform(X)
Z = linkage(X, method='ward')

# Method 1: cut by number of clusters
labels_3 = fcluster(Z, t=3, criterion='maxclust')

# Method 2: cut by distance threshold — find largest gap
merge_distances = Z[:, 2]
gaps = np.diff(merge_distances)
elbow_idx = np.argmax(gaps)
threshold = (merge_distances[elbow_idx] + merge_distances[elbow_idx + 1]) / 2
labels_auto = fcluster(Z, t=threshold, criterion='distance')
print(f"Auto threshold: {threshold:.3f} → {len(np.unique(labels_auto))} clusters")

# Annotated dendrogram
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=20, leaf_rotation=45)
ax.axhline(y=threshold, color='red', linestyle='--', label=f'Cut at {threshold:.2f}')
ax.set_title('Ward Dendrogram with Automatic Cut')
ax.set_xlabel('Cluster / Sample Index')
ax.set_ylabel('Merge Distance')
ax.legend()
plt.tight_layout()
plt.savefig('dendrogram_cut.png', dpi=100)
plt.show()
```

---

## 6. Cophenetic Correlation Coefficient

The **cophenetic correlation** measures how faithfully the dendrogram preserves pairwise distances. Values closer to 1 indicate a better-fitting hierarchy.

```python
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.6, random_state=42)
X = StandardScaler().fit_transform(X)

results = {}
for method in ['single', 'complete', 'average', 'ward']:
    Z = linkage(X, method=method)
    c, _ = cophenet(Z, pdist(X))
    results[method] = round(c, 4)

print("Cophenetic Correlation Coefficients:")
for m, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {m:10s}: {v}")
# Average and complete linkage typically score highest
```

**Interpretation:**
- c > 0.75: Good hierarchical fit.
- c < 0.65: Hierarchy distorts true distances; consider different linkage.
- Ward often has lower cophenetic correlation despite producing visually cleaner clusters — it optimises a different criterion.

---

## 7. sklearn AgglomerativeClustering

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
X = StandardScaler().fit_transform(X)

agg = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward',          # 'single', 'complete', 'average', 'ward'
    metric='euclidean',      # Ward requires Euclidean
    compute_full_tree=True,
    compute_distances=True   # Needed if accessing .distances_
)
labels = agg.fit_predict(X)
print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
print(f"n_clusters found: {agg.n_clusters_}")
```

**Note:** For non-Euclidean metrics (e.g., cosine, precomputed), use `linkage='average'` or `'complete'` — Ward requires Euclidean.

---

## 8. When to Use Hierarchical vs K-Means

| Criterion | Hierarchical | K-Means |
|---|---|---|
| **k known?** | Not required | Required |
| **Dataset size** | Small–medium (n ≤ 10 000) | Large (n > 10 000) |
| **Cluster relationships** | Reveals hierarchy/taxonomy | Flat partition only |
| **Shape** | Flexible (with right linkage) | Spherical |
| **Reproducibility** | ✅ Deterministic | ❌ Depends on init |
| **Memory** | O(n²) — expensive | O(n·k·d) |
| **Interpretability** | High — dendrogram | Moderate |
| **Use case** | Gene expression, document taxonomy, customer hierarchy | Large-scale segmentation |

---

## 9. Interview Q&A

**Q1: What is the difference between agglomerative and divisive hierarchical clustering?**  
A: Agglomerative (bottom-up) starts with n singleton clusters and merges the two closest at each step — O(n² log n). Divisive (top-down) starts with one cluster and recursively splits — O(2ⁿ) exhaustive search. Agglomerative is almost always used in practice due to computational feasibility.

**Q2: What is Ward's linkage and why is it preferred?**  
A: Ward's linkage merges the pair of clusters that produces the smallest increase in total within-cluster sum of squares (WCSS). It tends to create compact, equally-sized clusters and is the most robust general-purpose choice for numerical data. Limitation: requires Euclidean distance.

**Q3: What are the drawbacks of single linkage?**  
A: Single linkage is prone to the **chaining effect** — a sequence of close neighbouring points can link very distant clusters together via a chain. This produces elongated, non-compact clusters that often don't match the true structure.

**Q4: How do you decide where to cut the dendrogram?**  
A: (1) Visually identify the longest vertical lines (largest gaps) and cut there. (2) Compute the gap between consecutive merge distances; the largest gap suggests the optimal cut. (3) Use `fcluster` with `criterion='maxclust'` or `'distance'`. Always validate with Silhouette score.

**Q5: What is the cophenetic correlation and when is it low?**  
A: Cophenetic correlation measures how well the dendrogram preserves original pairwise distances. It's low when the linkage distorts distances — e.g., Ward focuses on variance minimisation, not distance preservation. High cophenetic (> 0.75) indicates the hierarchy is a faithful summary of the data geometry.

**Q6: Can hierarchical clustering handle non-Euclidean metrics?**  
A: Yes — single, complete, and average linkage work with any distance matrix (precomputed). Provide `metric='precomputed'` and a distance matrix to sklearn. Ward's linkage requires Euclidean distance.

**Q7: How does hierarchical clustering scale with n?**  
A: Naively O(n³); with an optimised implementation using priority queues, O(n² log n). The O(n²) memory requirement for the distance matrix makes it impractical for n > ~10 000. For large n, use Mini-Batch K-Means or BIRCH (scalable hierarchical alternative).

---

## 10. Common Pitfalls

1. **Not scaling features** — linkage distances are affected by feature scale; always StandardScale first.
2. **Using Ward with non-Euclidean distances** — Ward requires Euclidean; use average/complete otherwise.
3. **Over-relying on visual dendrogram cuts** — subjective; always validate with Silhouette score.
4. **Ignoring chaining with single linkage** — single linkage rarely gives meaningful clusters on noisy data.
5. **Trying hierarchical clustering on large n** — O(n²) memory blows up; switch to BIRCH or K-Means.
6. **Confusing merge height with cluster quality** — height measures merge distance, not cluster purity.
7. **Forgetting that labels are 1-indexed in scipy** — `fcluster` returns labels starting at 1, not 0.

---

## 11. Quick Reference Cheat Sheet

```
Algorithm:   Agglomerative (bottom-up) or Divisive (top-down)
Linkage methods:
  single   — min distance (chaining)
  complete — max distance (compact)
  average  — mean distance (robust)
  ward     — min WCSS increase (BEST for numerical data)
Choose k:    Cut dendrogram at largest gap; validate with Silhouette
Metric:      Any for single/complete/average; Euclidean ONLY for Ward
Memory:      O(n²) — unsuitable for n > ~10 000
Cophenetic:  > 0.75 = good hierarchy fit
Reproducible: YES — fully deterministic

scipy:
  from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
  Z      = linkage(X, method='ward')
  labels = fcluster(Z, t=4, criterion='maxclust')   # cut at k=4
  dendrogram(Z)

sklearn:
  from sklearn.cluster import AgglomerativeClustering
  agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
  labels = agg.fit_predict(X)
```
