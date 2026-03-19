# Imbalanced Data — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | Datasets where one class vastly outnumbers another (e.g., 99:1 ratio), causing models to be biased toward the majority class |
| **WHY** | Standard training minimizes overall loss — majority class dominates; minority class (often the important one) gets ignored |
| **WHEN** | Fraud detection, medical diagnosis, anomaly detection, churn prediction — any rare event problem |
| **HOW** | Class weights, resampling (SMOTE/ADASYN), threshold tuning, cost-sensitive learning |
| **PITFALLS** | SMOTE on full data before CV (leakage!), using accuracy as metric, oversampling test set |

---

## 1. Why Accuracy Fails — The 99% Accuracy Paradox

```python
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 1000 samples: 990 negative, 10 positive (1% positive class)
y_true = np.array([0] * 990 + [1] * 10)

# "Dumb" model: always predict majority class
y_pred_dumb = np.zeros(1000, dtype=int)

print("=== Always-Negative Model ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_dumb):.2%}")   # 99.0%
print(f"Recall:    {recall_score(y_true, y_pred_dumb):.2%}")     # 0.0%
print(f"F1 Score:  {f1_score(y_true, y_pred_dumb):.2%}")         # 0.0%
# Conclusion: 99% accuracy = completely useless model for minority class!
```

**The key insight:** With 1% positive class, predicting all-negative gives 99% accuracy. This model detects **zero** fraud, cancers, or failures. Always use F1, PR-AUC, or MCC for imbalanced problems.

---

## 2. Class Weights — Simplest Fix

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

X, y = make_classification(
    n_samples=10000, n_features=20, weights=[0.97, 0.03],
    random_state=42, n_informative=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Class distribution — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

# Without class weights
lr_default = LogisticRegression(max_iter=1000, random_state=42)
lr_default.fit(X_train, y_train)
print("\n--- Default (no class weights) ---")
print(classification_report(y_test, lr_default.predict(X_test), digits=3))

# With class_weight='balanced'  →  weight_i = n_samples / (n_classes × n_i)
lr_balanced = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=42
)
lr_balanced.fit(X_train, y_train)
print("\n--- class_weight='balanced' ---")
print(classification_report(y_test, lr_balanced.predict(X_test), digits=3))

# Custom weights
lr_custom = LogisticRegression(
    class_weight={0: 1, 1: 20},  # Penalize FN for class 1 by 20×
    max_iter=1000, random_state=42
)
lr_custom.fit(X_train, y_train)
print("\n--- Custom class weights {0:1, 1:20} ---")
print(classification_report(y_test, lr_custom.predict(X_test), digits=3))

# Tree-based models also support class_weight
rf_balanced = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42
)
# XGBoost: scale_pos_weight = n_negative / n_positive
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
print(f"\nXGBoost scale_pos_weight: {n_neg / n_pos:.1f}")
```

---

## 3. SMOTE — Synthetic Minority Over-Sampling Technique

SMOTE generates **synthetic** minority class samples by interpolating between existing minority samples and their k-nearest neighbors.

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
import numpy as np

smote = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)
# sampling_strategy=0.3 → minority becomes 30% of majority count

X_res_smote, y_res_smote = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {np.bincount(y_train)}")
print(f"After  SMOTE: {np.bincount(y_res_smote)}")
```

### ADASYN — Adaptive Synthetic Sampling

ADASYN generates more samples for minority class samples that are **harder to classify** (closer to the decision boundary).

```python
adasyn = ADASYN(sampling_strategy=0.3, n_neighbors=5, random_state=42)
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X_train, y_train)
print(f"After ADASYN: {np.bincount(y_res_adasyn)}")
```

### Combined: SMOTE + Tomek Links

```python
# SMOTE oversamples minority, then Tomek Links removes ambiguous majority borderline samples
smote_tomek = SMOTETomek(sampling_strategy=0.3, random_state=42)
X_res_combined, y_res_combined = smote_tomek.fit_resample(X_train, y_train)
print(f"After SMOTETomek: {np.bincount(y_res_combined)}")
```

**SMOTE variants:**

| Method | Strategy | Best For |
|--------|----------|----------|
| SMOTE | Interpolate between minority neighbors | General use |
| BorderlineSMOTE | Focus on borderline minority samples | Hard decision boundaries |
| ADASYN | More synthesis near hard samples | Complex boundaries |
| SMOTE+Tomek | Oversample minority + clean majority | Balanced + cleaner boundary |
| RandomUnderSampler | Randomly remove majority samples | Very large datasets |

---

## 4. Threshold Tuning

```python
from sklearn.metrics import (
    precision_recall_curve, f1_score, recall_score,
    precision_score, average_precision_score
)
import matplotlib.pyplot as plt

# Train model with balanced weights
lr_balanced.fit(X_train, y_train)
y_prob = lr_balanced.predict_proba(X_test)[:, 1]

# Sweep thresholds
thresholds = np.arange(0.01, 0.99, 0.01)
f1s, recalls, precisions = [], [], []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))

best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.2f}")
print(f"  F1:        {f1s[best_idx]:.4f}")
print(f"  Precision: {precisions[best_idx]:.4f}")
print(f"  Recall:    {recalls[best_idx]:.4f}")

# Find threshold for Recall ≥ 0.9 with max Precision
recall_arr = np.array(recalls)
precision_arr = np.array(precisions)
mask = recall_arr >= 0.9
if mask.any():
    best_recall_thresh_idx = np.argmax(precision_arr[mask])
    idx = np.where(mask)[0][best_recall_thresh_idx]
    print(f"\nHighest precision with Recall≥0.9: t={thresholds[idx]:.2f}, "
          f"P={precision_arr[idx]:.3f}, R={recall_arr[idx]:.3f}")
```

---

## 5. imbalanced-learn Pipeline — SMOTE on Train Only!

**Critical rule:** SMOTE must NEVER be applied to the test set. Synthetic samples only belong in the training fold.

```python
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn's Pipeline!
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np

# ✅ CORRECT: imbalanced-learn Pipeline automatically applies SMOTE
#             only to each training fold during cross-validation
pipe_smote = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),  # Applied per fold
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(
    pipe_smote, X, y, cv=cv,
    scoring=["f1", "roc_auc", "average_precision"],
    return_train_score=False
)

print("=== Correct SMOTE inside Pipeline ===")
for metric, scores in results.items():
    if metric.startswith("test_"):
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# ❌ WRONG: Applying SMOTE before CV — test folds see synthetic samples
X_wrong, y_wrong = SMOTE(random_state=42).fit_resample(X, y)
# Now doing CV on X_wrong, y_wrong is WRONG — leakage!
print("\n❌ The wrong way: SMOTE before CV inflates scores artificially")
```

---

## 6. Performance Comparison Table

```python
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, threshold=0.5):
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model": name,
        "F1":        round(f1_score(y_te, y_pred), 3),
        "Recall":    round(recall_score(y_te, y_pred), 3),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 3),
        "ROC-AUC":   round(roc_auc_score(y_te, y_prob), 3),
        "PR-AUC":    round(average_precision_score(y_te, y_prob), 3),
        "MCC":       round(matthews_corrcoef(y_te, y_pred), 3),
    }

results_list = []

# 1. Baseline: default LR
results_list.append(evaluate_model(
    "LR (default)", LogisticRegression(max_iter=1000, random_state=42),
    X_train, y_train, X_test, y_test
))

# 2. Class weights
results_list.append(evaluate_model(
    "LR (balanced)", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    X_train, y_train, X_test, y_test
))

# 3. SMOTE
X_sm, y_sm = SMOTE(sampling_strategy=0.3, random_state=42).fit_resample(X_train, y_train)
results_list.append(evaluate_model(
    "LR + SMOTE", LogisticRegression(max_iter=1000, random_state=42),
    X_sm, y_sm, X_test, y_test
))

# 4. RF balanced
results_list.append(evaluate_model(
    "RF (balanced)", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                             random_state=42),
    X_train, y_train, X_test, y_test
))

import pandas as pd
comparison_df = pd.DataFrame(results_list).set_index("Model")
print("\n=== Model Comparison on Imbalanced Dataset ===")
print(comparison_df.to_string())
```

---

## 7. Interview Q&A

**Q1: Why does accuracy fail for imbalanced datasets? Give a concrete example.**
> With 99% negative class, a model that predicts all-negative achieves 99% accuracy but 0% recall — it identifies zero positive cases. This is the accuracy paradox. For fraud detection where the positive class (fraud) is 1% of data, this "model" would miss every fraud case. Use F1, PR-AUC, or MCC instead.

**Q2: What is SMOTE and how does it work?**
> SMOTE (Synthetic Minority Over-Sampling Technique) generates synthetic minority class samples by interpolating between existing minority samples and their k-nearest neighbors. For each minority sample, it selects a random neighbor, draws a random point along the line segment between them, and adds it as a new synthetic sample. This avoids simple duplication (which just reweights) by creating new, diverse samples in the feature space.

**Q3: Why must SMOTE only be applied to training data?**
> SMOTE creates synthetic samples. If applied before cross-validation, synthetic copies of test-fold samples can appear in training folds, causing the model to see the test data — data leakage. This inflates performance estimates. The solution: use imbalanced-learn's Pipeline (not sklearn's Pipeline) which applies SMOTE only within each training fold during cross-validation.

**Q4: When would you use class weights vs SMOTE?**
> **Class weights:** Simplest approach, no risk of leakage, works well for moderate imbalance (up to ~10:1). Adjusts the loss function directly. **SMOTE:** Better for severe imbalance (>20:1); creates new samples which can help models learn better decision boundaries. However, SMOTE can create noisy samples in overlapping regions. When in doubt, try class weights first (simpler, less risk), then SMOTE if recall is still insufficient.

**Q5: What metrics should you use for imbalanced classification?**
> Avoid accuracy. Use: (1) **F1 score** — harmonic mean of precision and recall. (2) **PR-AUC** — area under the precision-recall curve, focuses on positive class. (3) **MCC** — uses all four confusion matrix cells, robust to imbalance. (4) **Recall** — if missing positives is very costly (fraud, disease). (5) **ROC-AUC** — good for ranking quality but can be misleading with severe imbalance (inflated by TN count).

**Q6: What is threshold tuning and when should you do it?**
> The default decision threshold (0.5) is rarely optimal for imbalanced data. Lowering the threshold increases recall (catch more positives) at the cost of precision (more false alarms). Threshold tuning: sweep thresholds on the validation set, compute the metric of interest (F1, or fixed-recall with max-precision), and select the optimal threshold. Never tune the threshold on the test set.

---

## 8. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Accuracy as metric** | 99% accuracy = 0% recall on imbalanced data | Use F1, PR-AUC, MCC |
| **SMOTE before CV** | Synthetic test samples leak into training | Use imblearn Pipeline |
| **SMOTE on test set** | Evaluating on synthetic data | Never resample test set |
| **Oversampling alone** | May not be enough for extreme imbalance | Combine SMOTE + class weights |
| **Ignoring threshold** | Default 0.5 is wrong for imbalanced data | Tune threshold on validation set |
| **No stratified splits** | Folds without positive samples | Always use StratifiedKFold |
| **SMOTE on raw features** | Interpolation meaningless for categoricals | Encode before SMOTE or use SMOTENC |

---

## 9. Quick Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│                  IMBALANCED DATA CHEAT SHEET                         │
├──────────────────┬───────────────────────────────────────────────────┤
│ STRATEGY         │ HOW TO USE                                        │
├──────────────────┼───────────────────────────────────────────────────┤
│ Class weights    │ class_weight="balanced" in sklearn models         │
│                  │ scale_pos_weight=neg/pos in XGBoost               │
│ SMOTE            │ SMOTE(sampling_strategy=0.3, random_state=42)     │
│                  │ MUST use imblearn Pipeline, not sklearn Pipeline  │
│ ADASYN           │ Focuses on hard-to-classify minority samples      │
│ SMOTETomek       │ SMOTE + remove Tomek Links (clean boundaries)     │
│ Threshold tuning │ Sweep on val set, pick by F1 or business rule     │
├──────────────────┴───────────────────────────────────────────────────┤
│ METRICS: Use F1 / PR-AUC / MCC — NOT accuracy                        │
│ SPLITS: Always StratifiedKFold                                        │
│ SMOTE: Only on training data, inside Pipeline                         │
│                                                                       │
│ Decision guide:                                                       │
│  Mild imbalance (2:1 to 10:1)  → class_weight="balanced"             │
│  Moderate (10:1 to 50:1)       → SMOTE + class_weight                │
│  Extreme (>50:1)               → SMOTE + ADASYN + threshold tuning   │
└──────────────────────────────────────────────────────────────────────┘
```
