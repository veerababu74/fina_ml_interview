# 🔀 Logistic Regression — Complete Interview Guide

> **5-Pillar Answer Framework:** WHAT | WHY | WHEN | HOW | PITFALLS

---

## 🏛️ The 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | A probabilistic classifier that models P(y=1\|x) using the sigmoid function applied to a linear combination of features |
| **WHY** | Outputs calibrated probabilities, interpretable log-odds coefficients, efficient, handles multi-class via softmax |
| **WHEN** | Binary/multi-class classification, need probability estimates, interpretability required, linear decision boundary sufficient |
| **HOW** | Maximizes log-likelihood via gradient descent; regularization with L1/L2; threshold tuning for precision/recall tradeoff |
| **PITFALLS** | Linear decision boundary (use polynomial features for non-linear), sensitive to outliers, requires feature scaling |

---

## 📐 Mathematical Foundation

### Sigmoid Function

The logistic (sigmoid) function maps any real value to (0, 1):

```
σ(z) = 1 / (1 + e⁻ᶻ)   where z = β₀ + β₁x₁ + ... + βₚxₚ = Xβ
```

Properties:
- σ(0) = 0.5 (decision boundary)
- σ(z) → 1 as z → +∞
- σ(z) → 0 as z → -∞
- Derivative: σ'(z) = σ(z)(1 - σ(z))

### Log-Odds (Logit) Interpretation

```
P(y=1|x) = σ(Xβ) = 1/(1 + e^(-Xβ))

log[P(y=1|x) / P(y=0|x)] = Xβ = β₀ + β₁x₁ + ... + βₚxₚ
```

Each coefficient β_j represents the change in **log-odds** for a one-unit increase in x_j. **Odds Ratio = e^(β_j)**.

### Binary Cross-Entropy Loss (Log-Loss)

```
L(β) = -1/n Σᵢ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
```

This is convex → guaranteed to find global minimum via gradient descent.

### Maximum Likelihood Estimation Derivation

Likelihood for n independent Bernoulli trials:
```
L(β) = Πᵢ p̂ᵢ^yᵢ (1-p̂ᵢ)^(1-yᵢ)
```

Log-likelihood (easier to maximize):
```
ℓ(β) = Σᵢ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
```

Minimizing negative log-likelihood = maximizing log-likelihood = minimizing cross-entropy.

Gradient:
```
∂ℓ/∂β = Xᵀ(y - p̂)   →   Update: β ← β + α·Xᵀ(y - p̂)
```

---

## 🌐 Multinomial Logistic Regression (Softmax)

For K classes, replace sigmoid with softmax:

```
P(y=k|x) = exp(Xβₖ) / Σⱼ exp(Xβⱼ)     for k = 1, ..., K
```

Properties:
- All probabilities sum to 1: Σₖ P(y=k|x) = 1
- Reduces to logistic regression for K=2
- Cross-entropy loss: L = -Σᵢ Σₖ yᵢₖ log(p̂ᵢₖ)

---

## 💻 Full Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss
)
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ── Generate Dataset ──────────────────────────────────────────────
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=2, class_sep=1.0, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 1. Basic Logistic Regression ─────────────────────────────────
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        C=1.0,              # Inverse regularization strength (1/λ)
        penalty='l2',       # Regularization type: 'l1', 'l2', 'elasticnet', None
        solver='lbfgs',     # Solver: 'lbfgs', 'liblinear', 'saga', 'newton-cg'
        max_iter=1000,
        class_weight=None,  # Set 'balanced' for imbalanced data
        random_state=42
    ))
])
lr_pipeline.fit(X_train, y_train)

# ── 2. Predictions ────────────────────────────────────────────────
y_pred = lr_pipeline.predict(X_test)
y_proba = lr_pipeline.predict_proba(X_test)[:, 1]

# ── 3. Comprehensive Metrics ──────────────────────────────────────
print("=" * 50)
print("LOGISTIC REGRESSION EVALUATION")
print("=" * 50)
print(classification_report(y_test, y_pred))
print(f"ROC-AUC:        {roc_auc_score(y_test, y_proba):.4f}")
print(f"Log-Loss:       {log_loss(y_test, y_proba):.4f}")
print(f"Brier Score:    {brier_score_loss(y_test, y_proba):.4f}")
print(f"Avg Precision:  {average_precision_score(y_test, y_proba):.4f}")

# ── 4. Cross-Validation ───────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nCV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Logistic Regression with CV (auto-tune C) ──────────────────
lr_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 20),   # C values to try
        cv=5,
        scoring='roc_auc',
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    ))
])
lr_cv.fit(X_train, y_train)
best_C = lr_cv.named_steps['model'].C_[0]
print(f"\nBest C (LR-CV): {best_C:.4f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, lr_cv.predict_proba(X_test)[:, 1]):.4f}")

# ── 6. Imbalanced Data with class_weight ──────────────────────────
lr_balanced = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        C=1.0, class_weight='balanced', solver='lbfgs',
        max_iter=1000, random_state=42
    ))
])
lr_balanced.fit(X_train, y_train)
print(f"\nBalanced LR ROC-AUC: {roc_auc_score(y_test, lr_balanced.predict_proba(X_test)[:, 1]):.4f}")

# ── 7. ROC Curve + Optimal Threshold ─────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Youden's J statistic: maximize TPR - FPR
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.4f}")
print(f"At optimal threshold:")
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))
```

---

## 📊 ROC Curve & Threshold Selection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


def plot_roc_and_pr_curves(y_true, y_proba, model_name="Model"):
    """Plot ROC and Precision-Recall curves with optimal thresholds."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── ROC Curve ──────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    # Optimal threshold: Youden's J
    j_idx = np.argmax(tpr - fpr)
    optimal_roc_threshold = roc_thresholds[j_idx]

    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0].scatter(fpr[j_idx], tpr[j_idx], s=100, c='red', zorder=5,
                    label=f'Optimal θ = {optimal_roc_threshold:.3f}')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve — {model_name}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Precision-Recall Curve ─────────────────────────────────────
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

    # F1-optimal threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    f1_idx = np.argmax(f1_scores)
    optimal_pr_threshold = pr_thresholds[f1_idx]

    axes[1].plot(recall, precision, 'g-', linewidth=2)
    axes[1].scatter(recall[f1_idx], precision[f1_idx], s=100, c='red', zorder=5,
                    label=f'Best F1 θ = {optimal_pr_threshold:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve — {model_name}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=100, bbox_inches='tight')
    print(f"Saved: roc_pr_curves.png")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Optimal ROC threshold: {optimal_roc_threshold:.4f}")
    print(f"Optimal F1 threshold: {optimal_pr_threshold:.4f}")

    return optimal_roc_threshold, optimal_pr_threshold
```

---

## ❓ Interview Questions & Answers

### Q1: Why can't we use MSE as the loss function for logistic regression?
**Answer:** MSE with sigmoid produces a **non-convex** loss landscape with many local minima, making optimization unreliable. Cross-entropy loss is convex for logistic regression → gradient descent always finds the global minimum. Additionally, MSE penalizes confident correct predictions (sigmoid saturates), while cross-entropy's log term provides strong gradients for wrong predictions.

### Q2: What does the C parameter in sklearn's LogisticRegression control?
**Answer:** C = 1/λ is the **inverse regularization strength**. Small C → strong regularization → simpler model → underfitting risk. Large C → weak regularization → complex model → overfitting risk. Default C=1.0. Compare: in Ridge/Lasso, α=λ directly (larger = more regularization). In LogisticRegression, C=1/λ (smaller = more regularization).

### Q3: When should you use ROC-AUC vs PR-AUC?
**Answer:** Use **ROC-AUC** when classes are roughly balanced — it measures overall discrimination ability across all thresholds. Use **PR-AUC** (Average Precision) when classes are imbalanced — ROC-AUC can be misleadingly optimistic because it includes true negatives in FPR calculation. PR curve focuses only on positive class performance.

### Q4: How does logistic regression handle multi-class classification?
**Answer:** Two strategies: (1) **One-vs-Rest (OvR/OvA)**: train K binary classifiers, each predicts P(y=k vs all others). (2) **Multinomial (Softmax)**: train one model with softmax output, all K probabilities sum to 1. Multinomial is theoretically better (joint training), OvR is faster (parallelizable). sklearn uses `multi_class='auto'` selecting based on solver.

### Q5: What is the interpretation of logistic regression coefficients?
**Answer:** β_j = change in **log-odds** per unit increase in x_j. More intuitively: **odds ratio = e^(β_j)**. Example: β_j = 0.5 → odds ratio = e^0.5 ≈ 1.65 → the odds of y=1 increase by 65% per unit increase in x_j. For standardized features, coefficients are comparable in magnitude.

### Q6: How do you assess model calibration in logistic regression?
**Answer:** Calibration = predicted probabilities match actual frequencies. Check with: (1) **Calibration curve** (reliability diagram) — plot mean predicted probability vs actual frequency per bin. (2) **Brier Score** = mean(p̂ᵢ - yᵢ)² — lower is better. (3) **Hosmer-Lemeshow test**. Logistic regression is often well-calibrated by default. If not, use `CalibratedClassifierCV`.

---

## ⚠️ Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Not scaling features | Slow convergence, poor performance | StandardScaler before fitting |
| Perfect separation (complete separation) | Coefficients → ∞, sklearn warning | Regularize (reduce C), check data |
| Ignoring class imbalance | High accuracy but bad recall on minority | `class_weight='balanced'`, PR-AUC metric |
| Using accuracy for imbalanced data | 95% accuracy with 95% negative class | Use F1, ROC-AUC, PR-AUC |
| Threshold = 0.5 always | Suboptimal precision/recall tradeoff | Tune threshold with Youden's J or F1 |
| Not checking multicollinearity | Unstable, uninterpretable coefficients | VIF check, use Ridge penalty |

---

## 📋 Quick Reference Cheat Sheet

```
Logistic Regression Quick Reference
══════════════════════════════════════════════════════════
Model:       P(y=1|x) = σ(Xβ) = 1/(1+e^(-Xβ))
Loss:        BCE = -1/n Σ[y·log(p̂) + (1-y)·log(1-p̂)]
Gradient:    ∂L/∂β = -Xᵀ(y - p̂)   (same form as linear!)

Multi-class: softmax P(y=k|x) = exp(Xβₖ)/Σⱼexp(Xβⱼ)

Solvers:
  lbfgs    → default, L-BFGS-B, good for L2, multi-class
  liblinear→ good for L1, small datasets
  saga     → L1/L2/ElasticNet, large datasets, stochastic

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
Threshold selection: 0.5 (default), Youden's J, F1-optimal

sklearn:
  LogisticRegression(C=1.0, penalty='l2', class_weight='balanced')
  LogisticRegressionCV(Cs=10, cv=5, scoring='roc_auc')
══════════════════════════════════════════════════════════
```
