# Anomaly Detection

## 1. Core Concept

Anomaly detection (also called outlier detection or novelty detection) identifies data points that deviate significantly from the expected pattern. Key distinction:

- **Outlier detection:** Train on data containing anomalies; model must distinguish them (unsupervised).
- **Novelty detection:** Train on clean data; detect new unseen patterns at inference (semi-supervised).

**Common applications:** Fraud detection, network intrusion, equipment failure, medical diagnosis, data quality.

---

## 2. Isolation Forest

### 2.1 Intuition — Path Length

Isolation Forest exploits the fact that **anomalies are few and different** — they are easy to isolate.

Algorithm:
```
1. Build T isolation trees (iTrees):
   a. Randomly select a feature.
   b. Randomly select a split value between min and max of that feature.
   c. Recursively partition until each point is isolated or max depth reached.

2. Compute path length h(x) for each point:
   h(x) = number of edges traversed to isolate point x

3. Anomaly score:
   s(x, n) = 2^{−E[h(x)] / c(n)}

   where c(n) = 2H(n−1) − 2(n−1)/n is the average path length for n samples
   and H(i) = ln(i) + 0.5772... (Euler-Mascheroni constant)
```

**Key insight:** Anomalies are isolated in **fewer splits** (shorter path = lower h(x) → score closer to 1). Normal points require many splits → score closer to 0.5.

| Score | Interpretation |
|-------|---------------|
| s → 1 | Almost certainly anomalous |
| s → 0.5 | No clear anomaly (normal) |
| s → 0 | Definitely normal (very deep isolation) |

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
# Normal data: 2D Gaussian
X_normal  = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], 300)
# Anomalies: uniform scattered far out
X_anom    = np.random.uniform(-6, 6, (20, 2))
X = np.vstack([X_normal, X_anom])
y_true = np.array([1]*300 + [-1]*20)  # -1 = anomaly

X_scaled = StandardScaler().fit_transform(X)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.06,  # expected fraction of anomalies
    max_samples='auto',
    random_state=42
)
y_pred = iso.fit_predict(X_scaled)       # 1=normal, -1=anomaly
scores  = iso.decision_function(X_scaled)  # higher = more normal

# Evaluate
from sklearn.metrics import classification_report
print("Isolation Forest Results:")
print(classification_report(y_true, y_pred, target_names=['anomaly','normal']))

# Visualise decision boundary
xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
Z = iso.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.5)
plt.colorbar(label='Anomaly Score (higher=normal)')
plt.scatter(X_scaled[y_true==1, 0],  X_scaled[y_true==1, 1],
            c='blue', s=10, label='Normal', alpha=0.5)
plt.scatter(X_scaled[y_true==-1, 0], X_scaled[y_true==-1, 1],
            c='red',  s=60, marker='*', label='Anomaly')
plt.title('Isolation Forest — Decision Boundary')
plt.legend()
plt.savefig('isolation_forest.png', dpi=100)
plt.show()
```

---

## 3. Local Outlier Factor (LOF)

### 3.1 Intuition — Local Density Comparison

LOF compares the **local density** of a point to the densities of its neighbours. A point in a sparse region surrounded by dense neighbours is an anomaly.

**Key definitions:**
```
k-distance(p):    distance to p's k-th nearest neighbour
reachability-distance_k(p, o) = max(k-distance(o), dist(p, o))
local reachability density:
  lrd_k(p) = 1 / (Σ_{o ∈ Nₖ(p)} reach-dist_k(p,o) / |Nₖ(p)|)
LOF_k(p) = (Σ_{o ∈ Nₖ(p)} lrd_k(o) / lrd_k(p)) / |Nₖ(p)|
```

**LOF interpretation:**
- LOF ≈ 1: Point has similar density to its neighbours → normal.
- LOF >> 1: Point is much less dense than its neighbours → anomaly.
- LOF < 1: Point is denser than neighbours (can occur in cluster centres).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.3],[0.3, 1]], 200)
X_anom   = np.array([[4, 4], [-4, -4], [3, -3], [-3, 3], [5, 0]])
X = np.vstack([X_normal, X_anom])
y_true = np.array([1]*200 + [-1]*5)

X_scaled = StandardScaler().fit_transform(X)

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.025,
    novelty=False  # novelty=True for out-of-sample prediction
)
y_pred = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous

from sklearn.metrics import classification_report
print("LOF Results:")
print(classification_report(y_true, y_pred, target_names=['anomaly','normal']))

# Plot with LOF score as bubble size
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=lof_scores, cmap='YlOrRd', s=lof_scores*20, alpha=0.6)
plt.colorbar(label='LOF Score (higher = more anomalous)')
plt.scatter(X_scaled[y_true==-1, 0], X_scaled[y_true==-1, 1],
            edgecolors='red', facecolors='none', s=100, linewidths=2, label='True Anomaly')
plt.title('Local Outlier Factor (LOF)')
plt.legend()
plt.savefig('lof.png', dpi=100)
plt.show()
```

---

## 4. One-Class SVM

### 4.1 Intuition — Boundary Learning

One-Class SVM learns a **boundary** around the normal data in a kernel-projected feature space. Points outside the boundary are anomalies.

```
Objective: Find a hyperplane that separates the origin from the majority of data
           in the kernel space with maximum margin.
Decision: f(x) = sign(w·φ(x) − ρ)  where φ is the kernel map
Score: w·φ(x) − ρ  (positive = normal, negative = anomaly)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_train = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], 300)
X_test_normal = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], 50)
X_test_anom   = np.random.uniform(-5, 5, (20, 2))
X_test = np.vstack([X_test_normal, X_test_anom])
y_test = np.array([1]*50 + [-1]*20)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

oc_svm = OneClassSVM(
    kernel='rbf',
    nu=0.05,    # upper bound on fraction of training errors (≈ contamination rate)
    gamma='scale'
)
oc_svm.fit(X_train_s)
y_pred   = oc_svm.predict(X_test_s)
scores   = oc_svm.decision_function(X_test_s)

from sklearn.metrics import classification_report
print("One-Class SVM Results:")
print(classification_report(y_test, y_pred, target_names=['anomaly','normal']))
```

**Choosing `nu`:** `nu` is an upper bound on the fraction of support vectors AND training errors. Set `nu` ≈ expected contamination rate (e.g., 0.05 for 5% outliers in training data).

---

## 5. Autoencoder for Anomaly Detection

### 5.1 Intuition — Reconstruction Error

An autoencoder trained on normal data learns a compressed representation of normal patterns. Anomalies, which were not seen during training, have high **reconstruction error**.

```
Architecture:
  Input → Encoder → Bottleneck (latent space) → Decoder → Reconstruction

Training:
  Minimise MSE(x, x̂) on normal data only.

Detection:
  reconstruction_error(x) = ‖x − x̂‖²
  Threshold: choose t such that recall/precision meets requirements.
  anomaly if reconstruction_error(x) > t
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Use pure numpy for simplicity (no deep learning framework required)
# In practice, use PyTorch or Keras

np.random.seed(42)

# Simulate: normal data follows a correlated multivariate Gaussian
X_normal = np.random.multivariate_normal(
    mean=[0]*5, cov=np.eye(5) + 0.3, size=500
)
X_anom = np.random.uniform(-6, 6, (30, 5))
X_all  = np.vstack([X_normal, X_anom])
y_all  = np.array([0]*500 + [1]*30)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Simple PCA-based autoencoder approximation (linear autoencoder = PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_reduced     = pca.fit_transform(X_scaled[:500])   # fit on normal only
X_reconstructed = pca.inverse_transform(pca.transform(X_scaled))
recon_error   = np.mean((X_scaled - X_reconstructed)**2, axis=1)

threshold = np.percentile(recon_error[:500], 97)  # 97th percentile of normal errors
y_pred = (recon_error > threshold).astype(int)

print("Linear Autoencoder (PCA) Results:")
print(classification_report(y_all, y_pred, target_names=['normal','anomaly']))
print(f"ROC-AUC: {roc_auc_score(y_all, recon_error):.3f}")

# Keras/PyTorch autoencoder sketch (not executed here)
KERAS_AUTOENCODER = """
import tensorflow as tf
from tensorflow.keras import layers, Model

input_dim = X_train.shape[1]

# Encoder
inputs   = layers.Input(shape=(input_dim,))
encoded  = layers.Dense(32, activation='relu')(inputs)
encoded  = layers.Dense(8,  activation='relu')(encoded)
bottleneck = layers.Dense(2, activation='relu')(encoded)   # latent dim=2

# Decoder
decoded  = layers.Dense(8,  activation='relu')(bottleneck)
decoded  = layers.Dense(32, activation='relu')(decoded)
outputs  = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

# Anomaly detection
recon = autoencoder.predict(X_test)
errors = np.mean((X_test - recon)**2, axis=1)
threshold = np.percentile(errors_train, 97)
anomaly_mask = errors > threshold
"""
print("\nKeras autoencoder sketch (see KERAS_AUTOENCODER variable)")
```

---

## 6. Full Comparison with Code

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.datasets import make_classification

np.random.seed(42)

# Dataset: imbalanced, 5% anomalies
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)
# Map: 0=normal(1), 1=anomaly(-1)
y_true = np.where(y == 0, 1, -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'Isolation Forest': IsolationForest(n_estimators=200, contamination=0.05, random_state=42),
    'LOF':              LocalOutlierFactor(n_neighbors=20, contamination=0.05),
    'One-Class SVM':    OneClassSVM(kernel='rbf', nu=0.05, gamma='scale'),
}

results = []
for name, model in models.items():
    if name == 'One-Class SVM':
        model.fit(X_scaled[y_true==1])  # fit on normal only
        y_pred  = model.predict(X_scaled)
        scores  = model.decision_function(X_scaled)
    else:
        y_pred  = model.fit_predict(X_scaled)
        if name == 'Isolation Forest':
            scores = model.decision_function(X_scaled)
        else:
            scores = -model.negative_outlier_factor_

    # Flip scores so higher = anomaly for ROC-AUC
    if name != 'LOF':
        scores_for_roc = -scores
    else:
        scores_for_roc = scores

    tp = ((y_pred == -1) & (y_true == -1)).sum()
    fp = ((y_pred == -1) & (y_true ==  1)).sum()
    fn = ((y_pred ==  1) & (y_true == -1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    auc  = roc_auc_score(y_true == -1, scores_for_roc)

    results.append({'Model': name, 'Precision': round(prec,3),
                    'Recall': round(rec,3), 'ROC-AUC': round(auc,3)})

df_results = pd.DataFrame(results).set_index('Model')
print(df_results.to_string())
```

### Algorithm Comparison Table

| | Isolation Forest | LOF | One-Class SVM | Autoencoder |
|--|--|--|--|--|
| **Type** | Tree ensemble | Density (kNN) | Kernel boundary | Deep learning |
| **Speed** | Fast O(n log n) | Slow O(n²) | Slow (kernel) | Fast inference |
| **High-d** | ✅ Handles well | ❌ Curse of dim | ❌ Kernel issues | ✅ Handles well |
| **New data** | ✅ predict() | ✅ novelty=True | ✅ predict() | ✅ predict() |
| **Interpretable** | Partial (paths) | Partial (scores) | ❌ | ❌ |
| **Params** | n_estimators, contamination | n_neighbors | nu, kernel | Architecture |
| **Best for** | General purpose, large n | Localised anomalies | Small-medium, clear boundary | Complex patterns, images |

---

## 7. Threshold Selection Strategy

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Given anomaly scores (higher = more anomalous) and true labels
np.random.seed(42)
scores = np.concatenate([np.random.normal(0, 1, 950),   # normal: low scores
                          np.random.normal(4, 1, 50)])    # anomalies: high scores
y_true = np.array([0]*950 + [1]*50)

precision, recall, thresholds = precision_recall_curve(y_true, scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx  = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best threshold: {best_threshold:.3f}")
print(f"Best F1:        {f1_scores[best_idx]:.3f}")
print(f"At P={precision[best_idx]:.3f}, R={recall[best_idx]:.3f}")
```

---

## 8. Interview Q&A

**Q1: How does Isolation Forest detect anomalies?**  
A: It builds random trees by recursively splitting features on random values. Anomalies — being few and distinct — are isolated in fewer splits (shorter average path length) than normal points. The anomaly score is 2^{−avg_path/c(n)}, where scores near 1 indicate anomalies and near 0.5 indicate normal points.

**Q2: What is the main advantage of LOF over Isolation Forest?**  
A: LOF detects **local** anomalies — points that are outliers relative to their local neighbourhood, not the global distribution. This is important when data has clusters of different densities. A point in a sparse cluster is not anomalous globally but may be anomalous locally.

**Q3: What is the `contamination` parameter in sklearn anomaly detectors?**  
A: It sets the proportion of training samples expected to be anomalies. This determines the decision threshold — the model flags the top `contamination` fraction of samples as anomalies. Set it to the known or estimated anomaly rate in your data.

**Q4: When should you use an autoencoder instead of statistical methods?**  
A: Use autoencoders when: (1) data has complex non-linear structure (images, time series), (2) statistical assumptions of other methods don't hold, (3) you have sufficient labelled normal data, (4) high-dimensional inputs like raw pixels or sensor streams. Downside: requires a neural network framework and hyperparameter tuning.

**Q5: How do you evaluate anomaly detection without labels?**  
A: (1) Silhouette score on the split between flagged and normal points. (2) Examine flagged anomalies manually or via business domain knowledge. (3) Inject known anomalies (synthetic) and measure recall. (4) Use reconstruction error distribution to check separation between normal and anomaly score distributions.

**Q6: Why is precision-recall AUC preferred over ROC-AUC for anomaly detection?**  
A: Anomaly detection datasets are highly imbalanced (e.g., 0.1% anomalies). ROC-AUC can be misleadingly high when TN is very large. Precision-Recall AUC focuses on the minority class performance, making it a more meaningful metric for imbalanced scenarios.

---

## 9. Common Pitfalls

1. **Treating anomaly detection as classification** — labelled data is usually unavailable; unsupervised approach is necessary.
2. **Not setting contamination correctly** — wrong contamination leads to too many or too few anomalies flagged.
3. **Using LOF on high-dimensional data** — kNN distances lose meaning in high-d; use Isolation Forest or autoencoders.
4. **Not scaling features** — distance-based methods (LOF, One-Class SVM) are sensitive to scale.
5. **Threshold by contamination alone** — always validate threshold on known anomaly examples or via domain knowledge.
6. **Evaluating only with accuracy** — accuracy is misleading when anomalies are < 1% of data; use F1, PR-AUC, ROC-AUC.
7. **Fitting on data that includes anomalies** — One-Class SVM and autoencoders should be fitted on clean normal data only.

---

## 10. Quick Reference Cheat Sheet

```
Isolation Forest:
  Principle:   Short path length in random trees = anomaly
  Params:      n_estimators=200, contamination=float
  Complexity:  O(n log n) — scales well
  Best for:    General purpose, high-dimensional, large datasets

LOF:
  Principle:   Low local density relative to neighbours = anomaly
  Params:      n_neighbors=20, contamination=float
  Complexity:  O(n²) — slow for large n
  Best for:    Localised / density-based anomalies

One-Class SVM:
  Principle:   Points outside learned boundary in kernel space = anomaly
  Params:      nu≈contamination, kernel='rbf', gamma='scale'
  Best for:    Novelty detection, clean training data available

Autoencoder:
  Principle:   High reconstruction error = anomaly
  Params:      Architecture, threshold (e.g., 97th percentile)
  Best for:    Images, time series, complex non-linear patterns

Evaluation:   ROC-AUC, PR-AUC, F1 (NOT accuracy)
Threshold:    Precision-recall curve + business requirements
Scaling:      MANDATORY for LOF and One-Class SVM

sklearn quick start:
  from sklearn.ensemble import IsolationForest
  iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
  y_pred = iso.fit_predict(X_scaled)  # 1=normal, -1=anomaly
  scores  = iso.decision_function(X_scaled)  # higher=more normal
```
