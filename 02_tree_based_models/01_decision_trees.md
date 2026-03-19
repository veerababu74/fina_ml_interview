# Decision Trees — Complete Interview Guide

## 5-Pillar Answer Framework

| Pillar | Answer |
|--------|--------|
| **WHAT** | A Decision Tree is a non-parametric supervised learning algorithm that recursively partitions the feature space into rectangular regions using axis-aligned splits, producing a tree of if-else rules. |
| **WHY** | It is interpretable (white-box), requires no feature scaling, handles mixed types, and captures non-linear relationships without transformation. |
| **WHEN** | Use when explainability is critical (e.g., credit decisions, medical diagnosis), as a baseline model, or as a building block inside ensemble methods (Random Forest, Gradient Boosting). |
| **HOW** | At each node, the algorithm greedily selects the feature and threshold that maximises the chosen splitting criterion (Gini/Entropy). Recursion stops when a stopping criterion is met (max depth, min samples, etc.). |
| **PITFALLS** | Prone to overfitting (high variance); unstable (small data changes produce very different trees); biased toward high-cardinality features when using raw impurity without correction. |

---

## Core Concept: How a Decision Tree Splits

```
Dataset (n samples, p features)
         |
         v
  ┌─────────────────────────────────────┐
  │   Root Node  (all n samples)        │
  │   Best split: X[2] <= 3.5 ?         │
  └──────────────┬──────────────────────┘
                 │
       ┌─────────┴──────────┐
       │ YES                │ NO
       v                    v
  ┌─────────┐          ┌─────────┐
  │ Node A  │          │ Node B  │
  │ n=60    │          │ n=40    │
  │ X[0]<=2?│          │ X[1]<=7?│
  └────┬────┘          └────┬────┘
       │                    │
   ┌───┴───┐            ┌───┴───┐
   │       │            │       │
   v       v            v       v
  Leaf   Leaf          Leaf   Leaf
 class=0 class=1      class=1 class=0
```

---

## Splitting Criteria: Gini vs Entropy

### Gini Impurity

Measures the probability that a randomly chosen element is incorrectly classified.

**Formula:**

```
Gini(t) = 1 - Σ p(c|t)²
               c
```

Where `p(c|t)` is the proportion of class `c` at node `t`.

**Binary classification example:**

```
p₀ = 0.7,  p₁ = 0.3
Gini = 1 - (0.7² + 0.3²)
     = 1 - (0.49 + 0.09)
     = 1 - 0.58
     = 0.42
```

- **Minimum (0):** Pure node — all samples belong to one class.
- **Maximum (0.5 for binary):** Perfectly mixed — 50/50 split.

### Entropy (Information-theoretic criterion)

Based on Shannon entropy; measures the expected number of bits to encode the class label.

**Formula:**

```
Entropy(t) = -Σ p(c|t) · log₂(p(c|t))
              c
```

**Binary classification example:**

```
p₀ = 0.7,  p₁ = 0.3
Entropy = -(0.7·log₂(0.7) + 0.3·log₂(0.3))
        = -(0.7·(-0.515) + 0.3·(-1.737))
        = -(−0.360 − 0.521)
        = 0.881 bits
```

- **Minimum (0):** Pure node.
- **Maximum (1 bit for binary):** Perfectly mixed.

### Which to choose?

| Property | Gini | Entropy |
|----------|------|---------|
| Computation | Faster (no log) | Slightly slower |
| Behaviour | Tends to isolate the largest class | More balanced splits |
| Sensitivity | Less sensitive to class probabilities | More sensitive |
| Default in sklearn | ✅ | Optional (`criterion='entropy'`) |

> **Interview tip:** Both criteria produce similar trees in practice. Choose Gini for speed and Entropy when you want to maximise information gain explicitly.

---

## Information Gain: Numerical Example

**Information Gain** measures how much a split reduces impurity:

```
IG(t, split) = Impurity(t) - Σ (|t_child| / |t|) · Impurity(t_child)
                              children
```

### Step-by-step example

**Dataset (14 samples, binary label):**
- Parent node: 9 positive (+), 5 negative (−)

```
Parent entropy:
p₊ = 9/14 ≈ 0.643,  p₋ = 5/14 ≈ 0.357
H(parent) = -(0.643·log₂(0.643) + 0.357·log₂(0.357))
           = -(0.643·(-0.637) + 0.357·(-1.486))
           = -(-0.409 - 0.530)
           = 0.940 bits
```

**Candidate split on feature X₁:**
- Left child (X₁ ≤ 3): 5 samples → 4+, 1−
- Right child (X₁ > 3): 9 samples → 5+, 4−

```
H(left)  = -(4/5·log₂(4/5) + 1/5·log₂(1/5))
          = -(0.8·(-0.322) + 0.2·(-2.322))
          = -(−0.258 − 0.464) = 0.722 bits

H(right) = -(5/9·log₂(5/9) + 4/9·log₂(4/9))
          = -(0.556·(-0.848) + 0.444·(-1.170))
          = -(−0.471 − 0.520) = 0.991 bits

Weighted entropy = (5/14)·0.722 + (9/14)·0.991
                 = 0.258 + 0.636 = 0.894 bits

IG(X₁, split=3) = 0.940 - 0.894 = 0.046 bits
```

The algorithm computes IG for all features and all thresholds, choosing the maximum.

---

## Stopping Criteria

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Hard limit on tree depth |
| `min_samples_split` | Node must have ≥ N samples to split |
| `min_samples_leaf` | Each child must have ≥ N samples |
| `min_impurity_decrease` | Split only if IG ≥ threshold |
| `max_leaf_nodes` | Limit total leaves (grows best-first) |

---

## Pruning Strategies

### Pre-pruning (Early Stopping)
Stop growing the tree before it becomes fully grown by enforcing stopping criteria during training. Fast but requires tuning thresholds.

### Post-pruning (Cost-Complexity Pruning)
Grow the full tree, then prune backward by removing subtrees that do not improve generalisation sufficiently.

**sklearn implementation — Cost-Complexity Pruning:**

```
R(T) = misclassification_rate(T) + α · |leaves(T)|
```

- `α` is the **ccp_alpha** hyperparameter (regularisation strength).
- Higher `α` → simpler tree (more pruning).
- Use cross-validation to find the optimal `α`.

---

## Bias-Variance Tradeoff vs Tree Depth

```
Error
  ↑
  │ Total Error
  │     ╲  ╱
  │      ╲╱
  │   ╱╲
  │  ╱  ╲ Bias²
  │ ╱    ╲___________
  │╱  Variance
  └─────────────────────→ Depth
  Shallow            Deep
```

| Depth | Bias | Variance | Overfit |
|-------|------|----------|---------|
| 1 (stump) | High | Low | No |
| Moderate | Medium | Medium | Balanced |
| Full (unlimited) | Low | High | Yes |

---

## Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# ── 1. Load data ──────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Baseline tree (no depth limit) ─────────────────────────────────────────
dt_full = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_full.fit(X_train, y_train)
print(f"Full tree depth : {dt_full.get_depth()}")
print(f"Full tree leaves: {dt_full.get_n_leaves()}")
print(f"Train accuracy  : {dt_full.score(X_train, y_train):.4f}")
print(f"Test  accuracy  : {dt_full.score(X_test,  y_test):.4f}")

# ── 3. Pruned tree (limited depth) ────────────────────────────────────────────
dt_pruned = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_pruned.fit(X_train, y_train)
print(f"\nPruned tree accuracy: {dt_pruned.score(X_test, y_test):.4f}")
print(classification_report(y_test, dt_pruned.predict(X_test),
                             target_names=class_names))

# ── 4. Cost-complexity pruning ────────────────────────────────────────────────
path = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[::5]          # subsample for speed

cv_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

best_alpha = ccp_alphas[np.argmax(cv_scores)]
print(f"\nBest ccp_alpha: {best_alpha:.6f}")

dt_ccp = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
dt_ccp.fit(X_train, y_train)
print(f"CCP tree accuracy: {dt_ccp.score(X_test, y_test):.4f}")

# ── 5. Feature importance ─────────────────────────────────────────────────────
importances = dt_pruned.feature_importances_
indices = np.argsort(importances)[::-1][:10]

print("\nTop-10 Feature Importances (MDI):")
for rank, idx in enumerate(indices, 1):
    print(f"  {rank:2d}. {feature_names[idx]:<35s} {importances[idx]:.4f}")

# ── 6. Hyperparameter grid search ─────────────────────────────────────────────
param_grid = {
    'max_depth': [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10, 20],
    'criterion': ['gini', 'entropy'],
}
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"\nBest params : {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.best_estimator_.score(X_test, y_test):.4f}")

# ── 7. Print text representation ──────────────────────────────────────────────
print("\nTree structure (depth=3):")
print(export_text(dt_pruned, feature_names=list(feature_names), max_depth=3))
```

---

## ASCII Art: Decision Tree Structure

```
                    ┌──────────────────────────────┐
                    │  ROOT NODE                   │
                    │  Feature: worst radius ≤ 16.8│
                    │  Gini: 0.468   Samples: 455  │
                    └──────────────┬───────────────┘
                                   │
                  ┌────────────────┴─────────────────┐
                  │ YES (≤ 16.8)                      │ NO (> 16.8)
                  v                                   v
    ┌──────────────────────────┐       ┌──────────────────────────┐
    │  INTERNAL NODE           │       │  INTERNAL NODE           │
    │  worst concave pts ≤ 0.14│       │  worst texture ≤ 25.7   │
    │  Gini: 0.282  n=306      │       │  Gini: 0.161  n=149      │
    └────────────┬─────────────┘       └──────────────┬───────────┘
                 │                                     │
         ┌───────┴───────┐                   ┌─────────┴──────────┐
         │ YES           │ NO                │ YES                │ NO
         v               v                  v                    v
   ┌──────────┐   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │   LEAF   │   │   LEAF   │        │   LEAF   │        │   LEAF   │
   │ class: B │   │ class: M │        │ class: B │        │ class: M │
   │ Gini:0.05│   │ Gini:0.42│        │ Gini:0.08│        │ Gini:0.10│
   │ n = 261  │   │ n =  45  │        │ n = 132  │        │ n =  17  │
   └──────────┘   └──────────┘        └──────────┘        └──────────┘

Legend: B = Benign, M = Malignant
```

---

## Interview Q&A

**Q1: Why do decision trees overfit easily?**

> They partition the training data until leaves are pure, memorising noise. A single tree with unlimited depth has essentially zero bias but very high variance. Remedies: pruning (`max_depth`, `ccp_alpha`), ensembling (Random Forest, Gradient Boosting).

**Q2: Gini vs Entropy — when does it matter?**

> In practice, the resulting trees are almost identical. Gini is faster (no log computation). Choose Entropy when you want splits that maximise information content explicitly; it can sometimes produce slightly more balanced trees.

**Q3: How does sklearn find the best split efficiently?**

> For each feature, it sorts the values (O(n log n)) and sweeps threshold candidates, computing impurity reduction in O(n). Total cost per node: O(p · n log n). Various optimisations (e.g., column sampling in `max_features`) reduce this further.

**Q4: What is cost-complexity pruning (ccp_alpha)?**

> After growing the full tree, it iteratively removes the weakest subtree — the one whose removal causes the smallest increase in misclassification rate per leaf removed — until the root is reached. This traces out a regularisation path parameterised by `α`. Cross-validate to choose the best `α`.

**Q5: Why are decision trees biased toward high-cardinality features?**

> Features with many unique values offer more threshold candidates, making it statistically more likely to find a split that reduces impurity by chance. Conditional inference trees or importance corrections (like those in `ranger` in R) address this. In practice, one-hot encoding high-cardinality categoricals before tree training mitigates the issue.

**Q6: What is the difference between `min_samples_split` and `min_samples_leaf`?**

> `min_samples_split` controls when a node is eligible to split (requires at least N samples in that node). `min_samples_leaf` enforces that **both** resulting children have at least N samples. The leaf constraint is a stricter regulariser because it prevents any split that would create a very small child.

**Q7: How are feature importances computed in a decision tree?**

> Mean Decrease in Impurity (MDI): for each feature, sum the weighted impurity reduction across all nodes where that feature was used as the split criterion. Normalise so importances sum to 1. **Limitation:** MDI is biased toward high-cardinality features and can be misleading when features are correlated. Permutation importance is a less biased alternative.

---

## Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| Severe overfitting | No depth limit | Set `max_depth`, use `ccp_alpha` |
| Unstable tree structure | High variance of single tree | Use Random Forest / Ensemble |
| Biased feature importance | MDI biased toward high-cardinality features | Use permutation importance or SHAP |
| Axis-aligned splits only | Algorithm design | Use oblique trees or kernel methods for rotated boundaries |
| Ignores feature interactions at same level | Greedy local search | Grow deeper or use ensembles |
| Poor on imbalanced data | Majority class dominates impurity | Use `class_weight='balanced'` |

---

## Quick Reference Cheat Sheet

```
DECISION TREE — QUICK REFERENCE
═══════════════════════════════════════════════════════════

SPLIT CRITERIA
  Gini     = 1 - Σ pᵢ²         (range: 0 to 0.5 binary)
  Entropy  = -Σ pᵢ log₂(pᵢ)   (range: 0 to 1 binary)
  IG       = H(parent) - Σ wᵢ·H(childᵢ)

KEY HYPERPARAMETERS
  max_depth            → primary depth regulariser
  min_samples_split    → min samples to consider splitting
  min_samples_leaf     → min samples in each leaf
  max_features         → number of features considered per split
  ccp_alpha            → cost-complexity pruning coefficient
  class_weight         → handles class imbalance

TYPICAL SKLEARN USAGE
  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier(max_depth=4, random_state=42)
  clf.fit(X_train, y_train)

COMPLEXITY
  Training : O(p · n · log n · depth)
  Prediction: O(depth)
  Space    : O(2^depth)

BIAS-VARIANCE
  Shallow tree → High Bias, Low Variance
  Deep tree   → Low Bias,  High Variance

WHEN TO USE
  ✅ Need interpretability / rules
  ✅ Mixed feature types
  ✅ As baseline or ensemble component
  ❌ High-dimensional sparse data
  ❌ Need top predictive performance (use ensemble instead)
═══════════════════════════════════════════════════════════
```
