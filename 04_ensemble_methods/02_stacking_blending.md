# Stacking and Blending

## 1. Core Concept

**Stacking** (stacked generalisation) trains a **meta-learner** on the predictions of multiple base models (level-0 learners). The meta-learner learns *how to best combine* the base model outputs, exploiting their complementary strengths.

**Key insight:** Instead of simple averaging (voting), a meta-learner discovers the optimal weighting — and can even learn non-linear combinations. Stacking almost always outperforms any single constituent model.

```
Level-0:  [Model A]  [Model B]  [Model C]   ← diverse base learners
              ↓          ↓          ↓
           pred_A     pred_B     pred_C
              ↓          ↓          ↓
Level-1:  [Meta-Learner] ← trained on [pred_A, pred_B, pred_C]
              ↓
        Final Prediction
```

---

## 2. The Data Leakage Problem

**Why naively training base models on full training data causes leakage:**

If we train base models on full X_train and then predict on X_train to get meta-features, the meta-learner sees predictions that were *trained on the same data*. These predictions will be unrealistically good (overfit), and the meta-learner learns to trust base model predictions that will not generalise.

**Solution: Out-of-Fold (OOF) Predictions** — the base models never see the data they predict on during meta-feature generation.

---

## 3. Out-of-Fold (OOF) Prediction Generation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(42)

# Dataset
X, y = make_classification(
    n_samples=1500, n_features=20, n_informative=12,
    n_redundant=4, random_state=42
)

# Train/test split (stacking: test set is untouched until final evaluation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
n_train = len(X_train)

# ------------------------------------------------------------------ #
#   OOF PREDICTION GENERATION — No Leakage!                          #
# ------------------------------------------------------------------ #
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Base learners
base_learners = {
    'lr':   LogisticRegression(max_iter=1000, random_state=42),
    'rf':   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'gb':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, random_state=42),
    'svm':  SVC(kernel='rbf', probability=True, random_state=42),
}

# Matrices to hold OOF predictions (meta-features for meta-learner)
oof_train = np.zeros((n_train, len(base_learners)))
# Test predictions: average over all K fold models
oof_test  = np.zeros((len(X_test), len(base_learners)))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

for col_idx, (name, model) in enumerate(base_learners.items()):
    oof_test_k = np.zeros((len(X_test), kf.n_splits))
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_s)):
        X_tr, X_val = X_train_s[tr_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[tr_idx],   y_train[val_idx]

        model.fit(X_tr, y_tr)

        # OOF prediction on validation fold
        oof_train[val_idx, col_idx] = model.predict_proba(X_val)[:, 1]

        # Test prediction for this fold
        oof_test_k[:, fold_idx] = model.predict_proba(X_test_s)[:, 1]

        fold_auc = roc_auc_score(y_val, oof_train[val_idx, col_idx])
        fold_scores.append(fold_auc)

    # Average test predictions across all K folds
    oof_test[:, col_idx] = oof_test_k.mean(axis=1)

    print(f"{name:5s}  OOF AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

print("\nOOF train meta-features shape:", oof_train.shape)
print("OOF test  meta-features shape:", oof_test.shape)
```

---

## 4. Meta-Learner Training

```python
# ------------------------------------------------------------------ #
#   LEVEL-1: Train meta-learner on OOF predictions                   #
# ------------------------------------------------------------------ #
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Simple logistic regression as meta-learner (often works best)
meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_learner.fit(oof_train, y_train)

# Final predictions on held-out test set
y_test_meta = meta_learner.predict(oof_test)
y_test_prob  = meta_learner.predict_proba(oof_test)[:, 1]

print("Stacking Results:")
print(f"  Test Accuracy: {accuracy_score(y_test, y_test_meta):.4f}")
print(f"  Test AUC:      {roc_auc_score(y_test, y_test_prob):.4f}")

# Compare with individual base learners (for reference)
print("\nIndividual base learner AUC on test set:")
for col_idx, name in enumerate(base_learners.keys()):
    auc = roc_auc_score(y_test, oof_test[:, col_idx])
    print(f"  {name:5s}: {auc:.4f}")
```

---

## 5. sklearn StackingClassifier

sklearn provides a clean implementation that handles OOF automatically:

```python
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = make_classification(n_samples=1500, n_features=20, n_informative=12,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define base estimators as (name, estimator) tuples
base_estimators = [
    ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
    ('rf',  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ('gb',  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=10)),
]

# Meta-learner
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# StackingClassifier with 5-fold OOF
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta,
    cv=5,                      # number of CV folds for OOF
    stack_method='predict_proba',  # use probability outputs as meta-features
    passthrough=False,         # set True to also pass original features to meta-learner
    n_jobs=-1
)

# Wrap in pipeline for scaling
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('stack',  stack)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print(f"StackingClassifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"StackingClassifier AUC:      {roc_auc_score(y_test, y_prob):.4f}")

# Cross-validation of the entire stacking pipeline
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 6. passthrough=True — Including Original Features

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# passthrough=True: meta-learner also receives original X as input
# Useful when some base learners might lose important information
stack_with_orig = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    passthrough=True,          # meta-learner sees [rf_prob, gb_prob, x₁, x₂, …]
    n_jobs=-1
)
```

---

## 7. Blending — Simpler Alternative

**Blending** is a simpler variant of stacking that uses a single holdout split instead of K-fold:

```
1. Split training data: 80% train_bl, 20% holdout
2. Train base learners on train_bl.
3. Predict on holdout → meta-features for meta-learner training.
4. Predict on test set → meta-features for final prediction.
5. Train meta-learner on holdout predictions.
6. Final prediction: meta-learner(test meta-features).
```

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

np.random.seed(42)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=12, random_state=42)

# Split: 70% for base training, 15% for blending holdout, 15% for final test
X_base, X_tmp,  y_base, y_tmp  = train_test_split(X, y, test_size=0.30, random_state=42)
X_hold, X_test, y_hold, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

scaler = StandardScaler()
X_base_s = scaler.fit_transform(X_base)
X_hold_s = scaler.transform(X_hold)
X_test_s = scaler.transform(X_test)

# Base learners
learners = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
]

hold_meta = []
test_meta  = []

for name, clf in learners:
    clf.fit(X_base_s, y_base)
    hold_meta.append(clf.predict_proba(X_hold_s)[:, 1])
    test_meta.append(clf.predict_proba(X_test_s)[:, 1])
    print(f"{name}: Holdout AUC = {roc_auc_score(y_hold, hold_meta[-1]):.4f}")

# Stack predictions into meta-feature matrices
X_hold_meta = np.column_stack(hold_meta)
X_test_meta  = np.column_stack(test_meta)

# Meta-learner
meta = LogisticRegression(max_iter=1000, random_state=42)
meta.fit(X_hold_meta, y_hold)
final_prob = meta.predict_proba(X_test_meta)[:, 1]

print(f"\nBlending Test AUC: {roc_auc_score(y_test, final_prob):.4f}")
```

---

## 8. Stacking vs Blending Comparison

| Property | Stacking (OOF) | Blending |
|---|---|---|
| **Data efficiency** | ✅ Uses all training data | ❌ Holds out 20-30% |
| **Leakage risk** | ✅ None (OOF prevents it) | ⚠️ Low (holdout is separate) |
| **Variance** | Lower (K full CV folds) | Higher (single holdout) |
| **Complexity** | Higher (K × base_learners fits) | Lower |
| **Training time** | K× longer per base learner | Simpler |
| **Competition use** | Preferred | Quick experiments |
| **Small datasets** | ✅ Better (uses all data) | ❌ Can't afford holdout |

---

## 9. Multi-Level Stacking

```python
# Level-2 stacking: stack the stackers
# Level-0: diverse base learners → OOF predictions A
# Level-1: intermediate stackers trained on A → OOF predictions B
# Level-2: final meta-learner trained on B

# Note: requires very careful cross-validation design to avoid leakage
# In practice, 2-level stacking is sufficient; 3+ levels rarely help

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

level1_stack = StackingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5, stack_method='predict_proba', n_jobs=-1
)

level2_stack = StackingClassifier(
    estimators=[
        ('stack1', level1_stack),
        ('knn',    KNeighborsClassifier(n_neighbors=15)),
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5, stack_method='predict_proba', n_jobs=-1
)
print("Multi-level stacking pipeline defined.")
```

---

## 10. Tips for Choosing Base Learners

Good stacking ensembles come from **diverse** base learners:

```python
# Diversity checklist:
# ✅ Different algorithms (tree-based + linear + distance-based)
# ✅ Different hyperparameters of the same algorithm
# ✅ Models trained on different feature subsets
# ✅ Models trained on different preprocessing (with/without PCA)
# ❌ Avoid: multiple nearly identical models (adds redundancy, not diversity)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

diverse_base_learners = [
    # Tree-based (high variance, low bias)
    ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
    ('et',  ExtraTreesClassifier(n_estimators=200, random_state=42)),
    # Linear (low variance, potentially high bias)
    ('lr',  LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    # Kernel
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    # Distance-based
    ('knn', KNeighborsClassifier(n_neighbors=15)),
    # Probabilistic
    ('nb',  GaussianNB()),
]
```

---

## 11. Interview Q&A

**Q1: What is the key difference between stacking and simple averaging?**  
A: Averaging assigns equal weight to all models. Stacking trains a meta-learner to find the *optimal* combination — it can weight models differently, combine them non-linearly, and exploit their complementary strengths. A meta-learner (e.g., Logistic Regression) can learn to trust the RF more for certain feature ranges and the SVM more for others.

**Q2: Why do we need OOF predictions? What happens without them?**  
A: Without OOF, base learners predict on data they trained on — their predictions are overfit (unrealistically good). The meta-learner then learns to trust these overfit predictions, which don't generalise. With OOF, each validation fold is never seen by the base learner during its training — predictions are honest estimates of out-of-sample performance.

**Q3: What is the difference between stacking and blending?**  
A: Both train a meta-learner on base model predictions. Stacking uses K-fold OOF to generate meta-features from all training data. Blending uses a single holdout split: base models train on 70-80% of data; their predictions on the remaining 20-30% form the meta-learner's training set. Stacking is more data-efficient and has lower variance; blending is simpler and faster to implement.

**Q4: What makes a good meta-learner?**  
A: Simple, regularised models work best: Logistic Regression, Ridge/Lasso, or a shallow GBDT. Complex meta-learners (deep trees, neural nets) tend to overfit the meta-features. The meta-learner's job is to learn *how to combine* the base model outputs, not to learn the original task again. Logistic Regression with its interpretable coefficients also tells you how much each base learner contributes.

**Q5: How does `passthrough=True` work and when is it useful?**  
A: With `passthrough=True`, the original training features are concatenated with the base model predictions as input to the meta-learner. This is useful when base learners may discard information present in the original features. However, it increases meta-learner input dimensionality and can lead to overfitting if not regularised. Use with regularised meta-learners (Lasso, Ridge).

**Q6: Can stacking overfit?**  
A: Yes, if OOF is not used correctly (leakage) or if the meta-learner is too complex. Also, if K is too small (e.g., K=2), OOF predictions have high variance. Mitigation: use K ≥ 5, use regularised meta-learners, and validate the entire stacking pipeline with a nested CV or a truly held-out test set.

**Q7: How do you implement stacking for regression?**  
A: Same principle — use `StackingRegressor` from sklearn or manual OOF. Base learners predict continuous values; meta-learner is typically Ridge or another linear model. OOF predictions are the predicted values (not probabilities).

---

## 12. Common Pitfalls

1. **Data leakage** — training base learners on all X_train and then using their predictions as meta-features without OOF. This is the most critical mistake in stacking.
2. **Non-diverse base learners** — five variants of Random Forest will not improve much over one; combine different algorithm families.
3. **Complex meta-learner** — a deep tree or neural net as meta-learner often overfits the OOF predictions; use regularised linear models.
4. **Forgetting test set scaling** — always fit scaler on training data only; transform test data with the same fitted scaler.
5. **Not re-fitting base learners on full training data** — for final predictions, sklearn's `StackingClassifier` re-fits each base learner on all training data; manual stacking should do the same.
6. **Using only 3 folds for OOF** — small K increases variance in OOF predictions; use K=5 or K=10.
7. **Evaluating each base learner instead of the stack** — the value of stacking comes from the combined output; evaluate the final stacked model.

---

## 13. Quick Reference Cheat Sheet

```
STACKING:
  Level-0: Train K diverse base learners on OOF folds
  OOF:     Each val fold is predicted by a model that never trained on it (no leakage)
  Level-1: Train meta-learner on OOF predictions as input features
  Test:    Base learners re-fitted on all train data → predict test → meta-learner
  Params:  cv=5, stack_method='predict_proba', passthrough=False

BLENDING:
  Simpler: single holdout (20-30%) for meta-feature generation
  Faster but higher variance; good for quick experiments

DIVERSITY CHECKLIST:
  ✅ Mix algorithm families: trees + linear + kernel + distance
  ✅ Different hyperparams
  ❌ Not 5 identical RFs

META-LEARNER:
  Best: LogisticRegression, Ridge, Lasso (regularised, simple)
  Avoid: Deep trees, neural nets (overfit meta-features)
  passthrough=True: also pass original features to meta-learner

sklearn quick start:
  from sklearn.ensemble import StackingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
  
  stack = StackingClassifier(
      estimators=[
          ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
          ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
      ],
      final_estimator=LogisticRegression(max_iter=1000, random_state=42),
      cv=5,
      stack_method='predict_proba',
      n_jobs=-1
  )
  stack.fit(X_train, y_train)
  y_pred = stack.predict(X_test)
```
