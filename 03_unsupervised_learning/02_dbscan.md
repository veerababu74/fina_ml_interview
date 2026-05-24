# DBSCAN — Density-Based Spatial Clustering of Applications with Noise

## 1. Core Concept

DBSCAN groups points that are **closely packed together** and marks points in low-density regions as **noise**. It discovers clusters of arbitrary shape without specifying k in advance.

Two hyperparameters:
- **ε (eps):** neighbourhood radius — how close two points must be to be "neighbours"
- **MinPts (min_samples):** minimum number of points within ε to form a dense region

---

## 2. Point Type Definitions

Given ε and MinPts, every point is classified as one of three types:

| Type | Definition | Role |
|------|-----------|------|
| **Core point** | Has ≥ MinPts neighbours within distance ε (including itself) | Anchors a cluster |
| **Border point** | Fewer than MinPts neighbours within ε, but within ε of a core point | Part of a cluster |
| **Noise point** | Not a core point and not within ε of any core point | Labelled −1 |

```
Visualisation:

       ★ core (MinPts=4, 5 neighbours within ε)
      ●●
     ●★●  ← border (only 2 neighbours within ε, but near a core)
      ●

    ○        ← noise (isolated, far from any core)
```

**Density-reachability:**
- Point q is **directly density-reachable** from core point p if dist(p,q) ≤ ε.
- Point q is **density-reachable** from p if there is a chain p → p₁ → … → q of direct density-reachable steps.
- Two points are **density-connected** if there exists a core point o from which both are density-reachable.

---

## 3. DBSCAN Algorithm — Step by Step

```
1. For each unvisited point p:
   a. Mark p as visited.
   b. Retrieve all points in N_ε(p) — the ε-neighbourhood.
   c. If |N_ε(p)| < MinPts: mark p as noise (may be reassigned later).
   d. Else (p is a core point):
      - Create new cluster C, add p to C.
      - Expand cluster:
          For each q in N_ε(p):
            If q not yet visited:
              Mark q as visited.
              If |N_ε(q)| ≥ MinPts: add N_ε(q) to the seed set.
            If q not yet assigned to any cluster: add q to C.
2. Return cluster assignments (noise = −1).
```

**Time complexity:** O(n · log n) with spatial indexing (k-d tree or ball tree); O(n²) naively.  
**Space complexity:** O(n) for the cluster assignments.

---

## 4. Tuning ε — K-Distance Graph

The k-distance graph plots, for each point, its distance to its **k-th nearest neighbour** (sorted in descending order). The "elbow" suggests a good ε.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Generate crescent-shaped clusters (impossible for K-Means)
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X = StandardScaler().fit_transform(X)

# k-distance graph — use k = min_samples - 1
k = 4  # will set min_samples=5
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, -1])[::-1]  # sort descending

plt.figure(figsize=(7, 4))
plt.plot(k_distances)
plt.axhline(y=0.25, color='red', linestyle='--', label='ε ≈ 0.25')
plt.xlabel('Points (sorted by k-distance)')
plt.ylabel(f'{k}-th Nearest Neighbour Distance')
plt.title('K-Distance Graph for DBSCAN ε Selection')
plt.legend()
plt.tight_layout()
plt.savefig('dbscan_k_distance.png', dpi=100)
plt.show()
print("Suggested ε (elbow region):", round(np.percentile(k_distances, 90), 3))
```

---

## 5. min_samples Rule of Thumb: 2 × d

For a dataset with **d features**, set:
```
min_samples ≥ 2 × d
```

Intuition: In d dimensions, a point needs at least 2d neighbours to reliably anchor a dense region. Additional heuristics:
- For **noisy data**: increase min_samples.
- For **large datasets** (n ≥ 100 000): min_samples ≥ ln(n).
- **Minimum:** min_samples = 3 (never < 3 for real data).

---

## 6. Full DBSCAN Example — Geospatial Clustering with Haversine

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Simulate GPS coordinates (latitude, longitude) for urban crime incidents
np.random.seed(42)
n = 600
# Three city hotspots + scattered noise
hotspot1 = np.random.normal(loc=[40.710, -74.006], scale=[0.01, 0.01], size=(200, 2))
hotspot2 = np.random.normal(loc=[40.750, -73.980], scale=[0.008, 0.008], size=(200, 2))
hotspot3 = np.random.normal(loc=[40.730, -74.000], scale=[0.015, 0.012], size=(150, 2))
noise    = np.column_stack([
    np.random.uniform(40.70, 40.77, 50),
    np.random.uniform(-74.02, -73.96, 50)
])
coords = np.vstack([hotspot1, hotspot2, hotspot3, noise])
df = pd.DataFrame(coords, columns=['lat', 'lon'])

# Haversine metric: distances in km
# Convert degrees to radians for haversine
coords_rad = np.radians(coords)

# eps in km / Earth_radius_km — 0.3 km radius
EARTH_RADIUS_KM = 6371.0
eps_km = 0.3
eps_rad = eps_km / EARTH_RADIUS_KM

db = DBSCAN(
    eps=eps_rad,
    min_samples=10,
    algorithm='ball_tree',
    metric='haversine'
)
df['cluster'] = db.fit_predict(coords_rad)

n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
n_noise    = (df['cluster'] == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points:   {n_noise} ({n_noise/len(df)*100:.1f}%)")
print(df.groupby('cluster').size().rename('count'))

# Plot
colors = plt.cm.tab10(np.linspace(0, 1, max(df['cluster']) + 1))
fig, ax = plt.subplots(figsize=(8, 6))
for cid in sorted(df['cluster'].unique()):
    mask = df['cluster'] == cid
    if cid == -1:
        ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                   c='black', s=10, marker='x', label='Noise', alpha=0.4)
    else:
        ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                   s=15, label=f'Cluster {cid}', alpha=0.6)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('DBSCAN Geospatial Clustering (Haversine metric)')
ax.legend()
plt.tight_layout()
plt.savefig('dbscan_geospatial.png', dpi=100)
plt.show()
```

---

## 7. Non-Spherical Shapes — Where DBSCAN Shines

```python
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, (X, title) in enumerate([
    (make_moons(300, noise=0.08, random_state=42)[0],   'Moons'),
    (make_circles(300, noise=0.05, factor=0.5, random_state=42)[0], 'Circles'),
]):
    X = StandardScaler().fit_transform(X)

    # K-Means
    km = KMeans(n_clusters=2, random_state=42)
    km_labels = km.fit_predict(X)
    axes[i][0].scatter(X[:, 0], X[:, 1], c=km_labels, cmap='bwr', s=10)
    axes[i][0].set_title(f'K-Means on {title}')

    # DBSCAN
    db = DBSCAN(eps=0.25, min_samples=5)
    db_labels = db.fit_predict(X)
    axes[i][1].scatter(X[:, 0], X[:, 1], c=db_labels, cmap='bwr', s=10)
    axes[i][1].set_title(f'DBSCAN on {title}')

plt.tight_layout()
plt.savefig('dbscan_vs_kmeans_shapes.png', dpi=100)
plt.show()
```

---

## 8. DBSCAN vs K-Means Comparison Table

| Property | DBSCAN | K-Means |
|---|---|---|
| **k specified?** | ❌ No | ✅ Yes (required) |
| **Cluster shape** | Arbitrary (any shape) | Spherical only |
| **Handles noise** | ✅ Built-in (label −1) | ❌ No (outliers corrupt centroids) |
| **Cluster density** | Same density required | Same density preferred |
| **Scalability** | O(n log n) with index | O(n·k·d·T) |
| **Deterministic** | ✅ Yes | ❌ Depends on init |
| **Varying densities** | ❌ Struggles (→ HDBSCAN) | ❌ Struggles |
| **High dimensions** | ❌ ε meaningless in high-d | ❌ Curse of dimensionality |
| **Params to tune** | ε, min_samples | k |
| **Best for** | Geospatial, irregular shapes, noise | Well-separated, convex blobs |

---

## 9. HDBSCAN — Handling Varying Densities

DBSCAN uses a single global ε, so it struggles when clusters have different densities. **HDBSCAN** (Hierarchical DBSCAN) extracts a cluster hierarchy and selects persistent clusters automatically.

```python
# pip install hdbscan
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
labels = clusterer.fit_predict(X)
print("HDBSCAN clusters:", len(set(labels)) - (1 if -1 in labels else 0))
```

---

## 10. Interview Q&A

**Q1: What is the difference between core, border, and noise points?**  
A: A **core point** has ≥ MinPts neighbours within ε — it anchors a cluster. A **border point** has fewer than MinPts neighbours but lies within ε of a core point — it belongs to that cluster. A **noise point** (outlier) is neither a core point nor within ε of any core point — labelled −1.

**Q2: How do you choose ε?**  
A: Use the k-distance graph: for each point compute the distance to its k-th nearest neighbour (k = min_samples − 1), sort descending, plot, and pick the ε at the "elbow" where the curve bends sharply. This marks where distances begin to increase rapidly — the boundary between core and noise.

**Q3: Why is DBSCAN deterministic but order-dependent for border points?**  
A: Core point assignments and noise labels are fully deterministic. However, a **border point** reachable from two different core points (in different clusters) will be assigned to whichever core point is processed first. In practice, this rarely affects results; use `algorithm='auto'` to ensure consistent ordering.

**Q4: Why does DBSCAN fail in high dimensions?**  
A: In high dimensions, all points become roughly equidistant (concentration of measure), making the concept of ε-neighbourhood meaningless — every point ends up with nearly the same distance to all others. Fix: reduce dimensionality with PCA first, or use cosine/Manhattan distance and retune ε.

**Q5: When should you use DBSCAN over K-Means?**  
A: Use DBSCAN when: (1) number of clusters is unknown, (2) clusters have irregular/non-spherical shapes, (3) data has significant noise/outliers, (4) working with geospatial data (use haversine metric). Use K-Means when clusters are roughly spherical, similar size, and k is known.

**Q6: How do you evaluate DBSCAN quality?**  
A: Silhouette Score (excluding noise points), DBCV (Density-Based Cluster Validation), or visual inspection. For geospatial, check if clusters correspond to meaningful geographic regions.

---

## 11. Common Pitfalls

1. **Not scaling features** — ε is in the same units as the features; unscaled data makes ε meaningless.
2. **Using too small min_samples** — setting `min_samples=2` creates spurious clusters from noise pairs; use ≥ 2d.
3. **Single global ε for varying-density data** — use HDBSCAN instead.
4. **Forgetting noise label −1 when computing metrics** — Silhouette requires excluding noise points.
5. **Running DBSCAN on raw high-dimensional data** — always reduce dimensions first.
6. **Interpreting cluster IDs as ordered** — DBSCAN cluster labels are arbitrary integers.
7. **Not using spatial indexing** — for large n, ensure `algorithm='ball_tree'` or `'kd_tree'` with an appropriate metric.

---

## 12. Quick Reference Cheat Sheet

```
Algorithm:   Density-based — expands clusters from core points
Key params:  ε (neighbourhood radius), min_samples (density threshold)
ε selection: k-distance graph elbow (k = min_samples − 1)
min_samples: Rule of thumb: 2 × n_features (minimum 3)
Noise:       Points labelled −1 (not assigned to any cluster)
Scaling:     MANDATORY — ε is distance-based
Metric:      Euclidean (default) or haversine for GPS
Shapes:      Handles arbitrary shapes — moons, rings, blobs
Fails for:   Varying densities (→ HDBSCAN), high dimensions
Time:        O(n log n) with spatial index, O(n²) naively

sklearn quick start:
  from sklearn.cluster import DBSCAN
  from sklearn.preprocessing import StandardScaler
  X_s = StandardScaler().fit_transform(X)
  db  = DBSCAN(eps=0.5, min_samples=5, algorithm='ball_tree')
  labels = db.fit_predict(X_s)
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  noise_mask = labels == -1
```
