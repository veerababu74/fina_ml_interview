# K-Means Clustering

## 1. Core Concept

K-Means partitions **n** observations into **k** clusters so each observation belongs to the cluster with the nearest mean (centroid). It minimises intra-cluster variance.

---

## 2. Lloyd's Algorithm — Step by Step

```
1. Choose k initial centroids  μ₁, μ₂, …, μₖ  (randomly or via K-Means++)
2. REPEAT until centroids stop moving:
   a. Assignment step  — assign every point xᵢ to its nearest centroid:
         cᵢ = argmin_j  ‖xᵢ − μⱼ‖²
   b. Update step      — recompute each centroid as the mean of assigned points:
         μⱼ = (1/|Cⱼ|) Σ_{i∈Cⱼ} xᵢ
3. Convergence when assignments no longer change (or change < tolerance)
```

**Convergence guarantee:** WCSS (see §3) decreases monotonically at every step, so the algorithm always converges — but may reach a local minimum.

**Time complexity:** O(n · k · d · iterations) where d = number of features.

---

## 3. WCSS Objective Function

**Within-Cluster Sum of Squares (Inertia):**

```
WCSS = Σⱼ Σ_{xᵢ ∈ Cⱼ} ‖xᵢ − μⱼ‖²
```

- Each step of Lloyd's algorithm reduces WCSS.
- WCSS always decreases as k increases → use the **Elbow Method** to pick k.
- WCSS is **not** comparable across datasets or different feature scalings.
- After StandardScaler, inertia values become interpretable in z-score units.

---

## 4. K-Means++ Initialization

Vanilla random initialization can lead to poor local minima.  
K-Means++ chooses seeds that are well-spread across the data:

```
1. Pick the first centroid uniformly at random from the data.
2. For each subsequent centroid:
   a. Compute D(xᵢ) = min distance from xᵢ to the nearest already-chosen centroid.
   b. Sample the next centroid with probability ∝ D(xᵢ)²
3. Repeat until k centroids are chosen, then run standard Lloyd's.
```

**Why it works:** Probability ∝ D² biases selection toward far-away points, spreading seeds across the feature space.  
**Benefit:** Expected WCSS is O(log k) times the optimum (Arthur & Vassilvitskii, 2007).  
**sklearn default:** `init='k-means++'` — always use this.

---

## 5. Choosing k — Elbow Method + Silhouette Score

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
X = StandardScaler().fit_transform(X)

inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Elbow plot
axes[0].plot(k_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters k')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=4, color='red', linestyle='--', label='Elbow at k=4')
axes[0].legend()

# Silhouette plot
axes[1].plot(k_range, sil_scores, 'gs-')
axes[1].set_xlabel('Number of Clusters k')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].axvline(x=4, color='red', linestyle='--', label='Best k=4')
axes[1].legend()

plt.tight_layout()
plt.savefig('kmeans_k_selection.png', dpi=100)
plt.show()
print(f"Best k by Silhouette: {k_range[np.argmax(sil_scores)]}")
```

**Silhouette score interpretation:**
| Score | Meaning |
|-------|---------|
| 0.71–1.0 | Strong structure |
| 0.51–0.70 | Reasonable structure |
| 0.26–0.50 | Weak structure |
| < 0.25 | No substantial structure |

---

## 6. Customer Segmentation — Complete Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- Simulate RFM (Recency, Frequency, Monetary) data ---
np.random.seed(42)
n = 1000
data = {
    'customer_id': range(1, n + 1),
    'recency':    np.random.exponential(scale=30, size=n).astype(int) + 1,   # days
    'frequency':  np.random.poisson(lam=5, size=n) + 1,                      # orders
    'monetary':   np.random.lognormal(mean=5, sigma=1, size=n).round(2),     # $ spent
}
df = pd.DataFrame(data)

# --- Preprocessing ---
features = ['recency', 'frequency', 'monetary']
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Fit K-Means with k=4 ---
km = KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=300, random_state=42)
df['segment'] = km.fit_predict(X_scaled)

# --- Analyse segments ---
segment_profile = df.groupby('segment')[features].agg(['mean', 'median'])
print("Segment Profiles:")
print(segment_profile.round(2))

# --- Label segments based on RFM logic ---
centers_original = scaler.inverse_transform(km.cluster_centers_)
segment_df = pd.DataFrame(centers_original, columns=features)
segment_df['label'] = segment_df.apply(
    lambda r: (
        'Champions'       if r['recency'] < 15  and r['frequency'] > 8  else
        'At Risk'         if r['recency'] > 60  and r['frequency'] < 3  else
        'Potential Loyal' if r['recency'] < 30  and r['frequency'] > 4  else
        'Hibernating'
    ), axis=1
)
print("\nCluster Centers (original scale):")
print(segment_df.round(2))

sil = silhouette_score(X_scaled, df['segment'])
print(f"\nSilhouette Score: {sil:.3f}")
print(f"Inertia: {km.inertia_:.1f}")

# --- Visualise (Recency vs Monetary, coloured by segment) ---
colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
fig, ax = plt.subplots(figsize=(8, 5))
for seg_id in range(4):
    mask = df['segment'] == seg_id
    ax.scatter(df.loc[mask, 'recency'], df.loc[mask, 'monetary'],
               c=colors[seg_id], alpha=0.5, label=segment_df.loc[seg_id, 'label'], s=15)
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Monetary ($)')
ax.set_title('Customer Segments (RFM K-Means)')
ax.legend()
plt.tight_layout()
plt.savefig('customer_segments.png', dpi=100)
plt.show()
```

---

## 7. Failure Cases and Fixes

| Failure Mode | Why It Fails | Fix |
|---|---|---|
| **Different cluster sizes** | Large clusters dominate centroid updates | Use GMM or DBSCAN |
| **Non-spherical (elongated) clusters** | WCSS assumes circular clusters | DBSCAN, Spectral Clustering |
| **Different cluster densities** | Dense and sparse clusters merged | DBSCAN (density-based) |
| **High-dimensional data** | Distance concentrates (curse of dimensionality) | PCA first, then K-Means |
| **Outliers / noise** | Single outlier pulls centroid far from mass | K-Medoids, remove outliers first |
| **Unscaled features** | High-variance features dominate ‖·‖ | Always `StandardScaler` first |
| **Local minimum** | Random restarts find different solutions | `n_init=20` or use K-Means++ |

```python
# Example: handling different cluster densities
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_classification
import numpy as np

# Simulate imbalanced clusters (one tight, one spread)
np.random.seed(42)
tight   = np.random.normal(loc=[0, 0],   scale=0.3, size=(50, 2))
spread  = np.random.normal(loc=[5, 5],   scale=2.0, size=(200, 2))
X_mixed = np.vstack([tight, spread])

# K-Means may split the spread cluster
km = KMeans(n_clusters=2, random_state=42)
labels_km = km.fit_predict(X_mixed)

# DBSCAN handles it naturally
db = DBSCAN(eps=0.8, min_samples=5)
labels_db = db.fit_predict(X_mixed)
print("K-Means unique labels:", np.unique(labels_km))
print("DBSCAN unique labels:", np.unique(labels_db))  # -1 = noise
```

---

## 8. Mini-Batch K-Means (Scalability)

```python
from sklearn.cluster import MiniBatchKMeans

# Trains on random mini-batches — much faster for large datasets
mbkm = MiniBatchKMeans(n_clusters=4, batch_size=512, n_init=10, random_state=42)
mbkm.fit(X_scaled)  # X_scaled from customer segmentation example
print(f"Mini-Batch Inertia: {mbkm.inertia_:.1f}")
```

- **Pros:** Scales to millions of samples.
- **Cons:** Slightly higher WCSS than full K-Means; not deterministic per se.

---

## 9. Interview Q&A

**Q1: What are the assumptions of K-Means?**  
A: (1) Clusters are spherical (isotropic), (2) clusters have similar sizes, (3) clusters have similar densities, (4) features are on the same scale. Violations degrade performance significantly.

**Q2: Why does K-Means always converge?**  
A: Each step (assignment + update) either decreases WCSS or leaves it unchanged. Since there are finitely many possible assignments, the algorithm must terminate. However, it may converge to a local minimum, not the global one.

**Q3: How does K-Means++ improve standard K-Means?**  
A: It samples the first centroid randomly, then samples each subsequent centroid with probability proportional to D(x)², where D(x) is the distance to the nearest existing centroid. This spreads seeds across the data and gives an expected solution within O(log k) of optimal.

**Q4: What is the Elbow Method and what are its limitations?**  
A: Plot WCSS vs k; the "elbow" where WCSS drops sharply then flattens suggests the best k. Limitation: the elbow is often ambiguous (no sharp bend). Supplement with Silhouette score, Gap Statistic, or domain knowledge.

**Q5: How does feature scaling affect K-Means?**  
A: K-Means uses Euclidean distance, so a feature with range [0, 10000] dominates one with range [0, 1]. Always standardise with `StandardScaler` before K-Means. Alternatively, use `MinMaxScaler` if features are bounded.

**Q6: Can K-Means handle categorical data?**  
A: No — Euclidean distance is undefined for categories. Alternatives: K-Modes (uses mode instead of mean), K-Prototypes (mixed), or encode categories carefully (e.g., binary encode then scale).

**Q7: What is the time complexity of K-Means?**  
A: O(n · k · d · T) where n = samples, k = clusters, d = dimensions, T = iterations. Usually T ≈ 100–300. It's linear in n, making it practical for large datasets. MiniBatchKMeans reduces the constant factor dramatically.

---

## 10. Common Pitfalls

1. **Forgetting to scale features** — distance-based, so unscaled features give garbage clusters.
2. **Running only one initialisation** — always set `n_init ≥ 10`.
3. **Choosing k by WCSS alone** — WCSS always decreases with k; use Silhouette or domain knowledge.
4. **Evaluating with labelled data as ground truth** — use ARI or NMI only to evaluate, not to select k (information leakage).
5. **Applying K-Means to non-convex cluster shapes** — use DBSCAN or Spectral Clustering for rings, moons, etc.
6. **Treating cluster IDs as ordinal** — cluster IDs are arbitrary labels (0, 1, 2 have no order).
7. **Ignoring outliers** — even a few extreme outliers heavily distort centroids.

---

## 11. Quick Reference Cheat Sheet

```
Algorithm:   Lloyd's (assign → update → repeat)
Objective:   Minimise WCSS = Σⱼ Σᵢ∈Cⱼ ‖xᵢ − μⱼ‖²
Init:        K-Means++ (default, use it always)
Restarts:    n_init=10 (sklearn default); set 20 for important runs
Scaling:     MANDATORY — StandardScaler before fit
Choose k:    Elbow (WCSS plot) + Silhouette score
Complexity:  O(n·k·d·T)  — linear in n
Assumptions: Spherical, similar-size, similar-density clusters
Fails for:   Non-convex shapes, outliers, very different densities
Alternatives:
  Non-spherical  → DBSCAN / Spectral Clustering
  Density-based  → DBSCAN
  Probabilistic  → GMM
  Scalable       → MiniBatchKMeans

sklearn quick start:
  from sklearn.cluster import KMeans
  km = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
  labels = km.fit_predict(X_scaled)
```
