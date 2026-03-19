# Classification Metrics — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | Quantitative measures that assess how well a classification model predicts discrete class labels |
| **WHY** | Accuracy alone is misleading (especially with imbalanced data); different metrics capture different failure modes |
| **WHEN** | Choose metric based on business cost: false positives vs false negatives, class balance, multi-class vs binary |
| **HOW** | Derive all metrics from the confusion matrix; tune decision threshold to shift the precision-recall tradeoff |
| **PITFALLS** | Accuracy paradox, threshold defaults at 0.5, data leakage into evaluation, macro vs micro confusion |

---

## 1. The Confusion Matrix

The confusion matrix is the foundation of all classification metrics.

```
                  Predicted Positive   Predicted Negative
Actual Positive  |       TP           |        FN         |
Actual Negative  |       FP           |        TN         |
```

**Definitions:**
- **TP (True Positive):**  Model predicted Positive, actual is Positive ✅
- **TN (True Negative):**  Model predicted Negative, actual is Negative ✅
- **FP (False Positive):** Model predicted Positive, actual is Negative ❌ (Type I Error)
- **FN (False Negative):** Model predicted Negative, actual is Positive ❌ (Type II Error)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
# [[TN, FP],
#  [FN, TP]]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

---

## 2. Core Metrics with Formulas

### Accuracy
> The fraction of all predictions that are correct.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Use when:** Classes are balanced and all errors carry equal cost.
**Avoid when:** Classes are imbalanced (99% negative class → 99% accuracy by predicting all-negative).

---

### Precision
> Of all positive predictions, how many were actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**High precision matters when:** False positives are costly.
**Example:** Spam filter — marking legitimate email as spam (FP) is bad.

---

### Recall (Sensitivity / True Positive Rate)
> Of all actual positives, how many did the model correctly identify?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**High recall matters when:** False negatives are costly.
**Example:** Cancer screening — missing a cancer case (FN) is catastrophic.

---

### F1 Score
> Harmonic mean of Precision and Recall.

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Why harmonic mean?** It punishes extreme imbalance — a model with precision=1.0 and recall=0.01 gets F1=0.02, not 0.5.

**F-beta Score** (generalized):
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- **β > 1:** Weights recall higher (critical detection tasks)
- **β < 1:** Weights precision higher (cost-sensitive tasks)

---

### Specificity (True Negative Rate)
$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Pair with Recall** to understand the full performance picture.

---

### ROC-AUC (Area Under the ROC Curve)

**ROC Curve:** Plots **True Positive Rate (Recall)** vs **False Positive Rate** at every threshold.

$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

**AUC Physical Meaning (Ranking Interpretation):**
> AUC = probability that the model ranks a random positive instance higher than a random negative instance.

- AUC = 1.0 → Perfect ranking
- AUC = 0.5 → Random guessing
- AUC = 0.0 → Perfect inverse ranking

This makes AUC **threshold-invariant** — it evaluates the model's discriminative ability across all thresholds.

---

### PR-AUC (Precision-Recall Area Under the Curve)

**PR Curve:** Plots **Precision** vs **Recall** at every threshold.

**PR-AUC vs ROC-AUC for Imbalanced Data:**

| Condition | Prefer |
|-----------|--------|
| Balanced classes | ROC-AUC |
| Highly imbalanced (rare positives) | PR-AUC |
| Care about ranking positives | PR-AUC |

**Why PR-AUC is better for imbalanced data:**
ROC-AUC can be inflated because the denominator of FPR includes the large TN count. With 99% negatives, even a model that produces many FPs can have a low FPR. PR-AUC focuses only on the positive class, exposing the real failure mode.

---

### Matthews Correlation Coefficient (MCC)

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

- Range: **[-1, +1]**
- **+1:** Perfect prediction
- **0:** Random prediction
- **-1:** Perfect inverse prediction

**Use case:** Best single metric for imbalanced binary classification — uses all four cells of the confusion matrix. Recommended by many researchers over F1 for binary classification.

---

## 3. Complete Python Code — All Metrics

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=1000, n_features=20, weights=[0.9, 0.1],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Metrics ===")
print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:        {average_precision_score(y_test, y_prob):.4f}")
print(f"MCC:           {matthews_corrcoef(y_test, y_pred):.4f}")

print("\n=== Detailed Report ===")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
```

---

## 4. Multi-Class Metrics (Macro / Micro / Weighted)

| Averaging | Description | Use When |
|-----------|-------------|----------|
| **Macro** | Unweighted mean of per-class scores | Each class equally important regardless of size |
| **Micro** | Aggregate TP/FP/FN across all classes | Large-scale datasets, prefer overall accuracy |
| **Weighted** | Mean weighted by class support (sample count) | Imbalanced multi-class, care about overall performance |

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

for avg in ["macro", "micro", "weighted"]:
    p = precision_score(y_test, y_pred, average=avg)
    r = recall_score(y_test, y_pred, average=avg)
    f = f1_score(y_test, y_pred, average=avg)
    print(f"{avg:>10} → Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}")
```

---

## 5. Threshold Tuning

The default decision threshold is **0.5**, but this is rarely optimal.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Sweep thresholds
thresholds = np.arange(0.1, 0.9, 0.05)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

# Find best F1 threshold
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
print(f"Best threshold by F1: {best_threshold:.2f} → F1={f1s[best_idx]:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label="Precision", marker="o")
plt.plot(thresholds, recalls, label="Recall", marker="s")
plt.plot(thresholds, f1s, label="F1", marker="^", linewidth=2)
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best t={best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend()
plt.grid(True)
plt.show()
```

**Threshold selection strategy:**
- **Maximize F1:** Balanced precision-recall tradeoff
- **Fix Recall ≥ 0.9:** Then pick highest precision (e.g., medical screening)
- **Fix Precision ≥ 0.95:** Then pick highest recall (e.g., fraud detection approval)

---

## 6. ROC and PR Curve Plots

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# PR Curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(True)

# PR
axes[1].plot(recall_curve, precision_curve, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", label=f"Baseline = {y_test.mean():.2f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

## 7. Interview Q&A

**Q1: Why not always use accuracy?**
> With 99% negative class, a model predicting all-negative achieves 99% accuracy but 0% recall. Accuracy is only meaningful when classes are balanced and all error types have equal cost.

**Q2: When would you choose Recall over Precision?**
> Choose recall when false negatives are more costly than false positives. Medical diagnosis (missing cancer = catastrophic), fraud detection (missing fraud = financial loss). Choose precision when false positives are more costly: spam filtering (blocking legitimate mail), hiring (filtering out good candidates).

**Q3: Explain the AUC-ROC physical meaning.**
> AUC equals the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance. It measures ranking quality across all decision thresholds, making it threshold-agnostic.

**Q4: Why use PR-AUC instead of ROC-AUC for imbalanced data?**
> ROC-AUC includes True Negatives in the FPR denominator. With massive class imbalance, TN is huge, making FPR artificially small even if the model produces many FPs relative to positives. PR-AUC focuses exclusively on the positive class (precision and recall), revealing the actual model quality for the minority class.

**Q5: What is MCC and when do you use it?**
> MCC (Matthews Correlation Coefficient) is a correlation coefficient between predicted and actual binary labels. Unlike F1, it uses all four cells of the confusion matrix and is robust to class imbalance. Prefer MCC when you want a single balanced metric for binary classification with imbalanced data.

**Q6: How do you pick the decision threshold?**
> The threshold should be selected based on the business objective, not left at 0.5. Use a validation set to sweep thresholds and select based on: maximizing F1 (balanced), fixing recall ≥ threshold (critical detection), or fixing precision ≥ threshold (precision-sensitive tasks). Never tune thresholds on the test set.

**Q7: What's the difference between macro, micro, and weighted averaging?**
> **Macro:** Average of per-class metrics, treating all classes equally (good when each class matters equally). **Micro:** Aggregates TP/FP/FN globally (dominated by majority class). **Weighted:** Weighted by class support — natural choice for imbalanced multi-class when you care about overall performance proportional to class frequency.

**Q8: How would you evaluate a model when both classes have very different misclassification costs?**
> Use cost-sensitive evaluation: compute a cost matrix where each cell (FP, FN, TP, TN) has an associated cost, then compute total expected cost. This is more direct than any single metric and aligns evaluation with business impact.

---

## 8. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Accuracy paradox** | 99% accuracy with 1% positive class | Use F1, PR-AUC, or MCC |
| **Default threshold** | Always using 0.5 | Tune threshold on validation set |
| **Test set threshold tuning** | Selecting threshold on test data | Use validation set or nested CV |
| **Macro vs micro confusion** | Using wrong averaging for multi-class | Understand class imbalance first |
| **Ignoring calibration** | Probabilities not meaningful | Use Brier score, calibration plots |
| **Leakage into evaluation** | Test data seen during training | Strict train/test separation |
| **Forgetting class order** | `predict_proba[:, 1]` vs `[:, 0]` | Always verify positive class index |

---

## 9. Quick Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────┐
│                 CLASSIFICATION METRICS CHEAT SHEET               │
├─────────────┬────────────────────────┬────────────────────────────┤
│ Metric      │ Formula                │ Use When                   │
├─────────────┼────────────────────────┼────────────────────────────┤
│ Accuracy    │ (TP+TN)/Total          │ Balanced classes           │
│ Precision   │ TP/(TP+FP)             │ FP cost is high            │
│ Recall      │ TP/(TP+FN)             │ FN cost is high            │
│ F1          │ 2·P·R/(P+R)            │ Balance P and R            │
│ ROC-AUC     │ P(score+ > score-)     │ Balanced, ranking quality  │
│ PR-AUC      │ Area under P-R curve   │ Imbalanced classes         │
│ MCC         │ (TP·TN-FP·FN)/denom    │ Best single metric overall │
├─────────────┴────────────────────────┴────────────────────────────┤
│ THRESHOLD: Lower → higher recall, lower precision                 │
│ THRESHOLD: Higher → higher precision, lower recall                │
│ AUC = 0.5 (random), AUC = 1.0 (perfect)                          │
│ MCC range: [-1, 0, +1] = [inverse, random, perfect]              │
└──────────────────────────────────────────────────────────────────┘
```

**Decision Guide:**
```
Is class imbalanced (>10:1)?
  YES → Use PR-AUC, F1, MCC (avoid accuracy)
  NO  → Accuracy or ROC-AUC work well

Is it multi-class?
  Each class equally important? → Macro average
  Dominated by large classes?  → Weighted average
  Want aggregate performance?  → Micro average

Is cost asymmetric?
  YES → Tune threshold; use cost matrix
  NO  → F1 or ROC-AUC
```
