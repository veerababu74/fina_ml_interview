# AdaBoost — Complete Interview Guide

## 5-Pillar Answer Framework

| Pillar | Answer |
|--------|--------|
| **WHAT** | AdaBoost (Adaptive Boosting) is an ensemble method that trains a sequence of **weak learners** (typically shallow decision trees), iteratively re-weighting training samples so that subsequent learners focus on misclassified examples. Final prediction is a weighted majority vote. |
| **WHY** | It converts weak learners into a strong classifier with exponentially fast convergence on training error, and its adaptive reweighting mechanism focuses model capacity where it is needed most. |
| **WHEN** | Use when base learners are fast to train (stumps), dataset is clean (AdaBoost is sensitive to noisy labels and outliers), and moderate accuracy with interpretability is required. |
| **HOW** | Maintain a distribution over training samples. At each round: train a weak learner on the current distribution; compute its weighted error; calculate its vote weight α; update sample weights to upweight misclassified examples; normalise. Final model is a weighted sum of weak learners. |
| **PITFALLS** | Highly sensitive to label noise and outliers (mislabelled samples get exponentially upweighted); requires weak learners to perform better than random; slower sequential training; may overfit with too many estimators on noisy data. |

---

## The AdaBoost Algorithm Step by Step

### Setup

- Training set: `{(x₁, y₁), ..., (xₙ, yₙ)}` where `yᵢ ∈ {-1, +1}`
- Number of rounds: `T`
- Base learner: typically a decision stump (depth-1 tree)

### Initialise Sample Weights

```
D₁(i) = 1/n    for all i = 1, ..., n
```

All samples start with equal weight.

### For each round t = 1, ..., T:

**Step 1: Train weak learner**
```
hₜ = argmin_{h ∈ H} Σᵢ Dₜ(i) · 𝟙[h(xᵢ) ≠ yᵢ]
```
Find the stump that minimises weighted training error.

**Step 2: Compute weighted error**
```
εₜ = Σᵢ Dₜ(i) · 𝟙[hₜ(xᵢ) ≠ yᵢ]
```
This is the weighted fraction of misclassified samples (εₜ < 0.5 required).

**Step 3: Compute learner weight (alpha)**
```
αₜ = ½ · ln((1 - εₜ) / εₜ)
```

Properties:
- If εₜ → 0 (perfect): αₜ → +∞ (huge vote)
- If εₜ = 0.5 (random): αₜ = 0 (no vote)
- If εₜ → 1 (always wrong): αₜ → -∞ (negative vote — flip its prediction)

**Step 4: Update sample weights**
```
Dₜ₊₁(i) = Dₜ(i) · exp(-αₜ · yᵢ · hₜ(xᵢ)) / Zₜ
```

Where `Zₜ` is a normalisation factor ensuring `Σᵢ Dₜ₊₁(i) = 1`.

Simplified:
```
If hₜ(xᵢ) = yᵢ (correct):   Dₜ₊₁(i) ∝ Dₜ(i) · e^{-αₜ}   (weight DECREASES)
If hₜ(xᵢ) ≠ yᵢ (wrong):     Dₜ₊₁(i) ∝ Dₜ(i) · e^{+αₜ}   (weight INCREASES)
```

### Final Prediction

```
H(x) = sign( Σₜ αₜ · hₜ(x) )
```

The final classifier is a weighted majority vote of all T weak learners.

---

## Alpha Calculation: Numerical Example

### Round 1 Example

**Setup:** 10 training samples, 5 positive (+1) and 5 negative (-1).

```
Initial weights: D₁(i) = 1/10 = 0.1 for all i

Sample:  x₁  x₂  x₃  x₄  x₅  x₆  x₇  x₈  x₉  x₁₀
Label:   +1  +1  +1  +1  +1  -1  -1  -1  -1  -1
D₁:      0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
```

**Best stump (h₁):** predicts +1 if x < threshold, else -1.
- Correctly classifies x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈
- Misclassifies x₉, x₁₀ (both are -1 but predicted +1)

**Weighted error:**
```
ε₁ = D₁(x₉) + D₁(x₁₀) = 0.1 + 0.1 = 0.2
```

**Alpha:**
```
α₁ = ½ · ln((1 - 0.2) / 0.2)
   = ½ · ln(0.8 / 0.2)
   = ½ · ln(4)
   = ½ · 1.386
   = 0.693
```

**Weight update:**
```
For correctly classified (x₁–x₈):
  D₂(i) ∝ 0.1 · e^{-0.693} = 0.1 · 0.5 = 0.05

For misclassified (x₉, x₁₀):
  D₂(i) ∝ 0.1 · e^{+0.693} = 0.1 · 2.0 = 0.20

Sum = 8 × 0.05 + 2 × 0.20 = 0.40 + 0.40 = 0.80
Normalised:
  Correct:      D₂(i) = 0.05/0.80 = 0.0625
  Misclassified: D₂(i) = 0.20/0.80 = 0.2500
```

Now x₉ and x₁₀ have 4× the weight of correctly classified samples, forcing h₂ to focus on them.

---

## Weight Update Visualisation

```
Round 1: Equal weights
  ●●●●●○○○○○    (● = positive, ○ = negative, same size)

Round 2: x₉, x₁₀ upweighted
  ●●●●●○○○◉◉    (◉ = upweighted — 4× larger)

Round 3: misclassified in round 2 get upweighted
  ...
```

---

## SAMME vs SAMME.R

AdaBoost was originally binary. For multi-class problems, sklearn implements two variants:

| Algorithm | Type | Update | Probability |
|-----------|------|--------|-------------|
| **SAMME** | Discrete | Uses class predictions | No |
| **SAMME.R** | Real-valued | Uses class probabilities | Yes |

### SAMME (Stagewise Additive Modelling using Multiclass Exponential loss)

Alpha formula for K classes:
```
αₜ = ln((1 - εₜ)/εₜ) + ln(K - 1)
```

### SAMME.R (Real-valued SAMME)

Uses the probability estimates from the weak learner. At each round, the update is:

```
f(x) += (K-1)/K · [log p(y|x) - (1/K) Σₖ log p(k|x)]
```

- **SAMME.R usually converges faster and achieves better accuracy** when base learners support probability outputs.
- `algorithm='SAMME.R'` is the default in sklearn.

---

## Staged Prediction & Learning Curves

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Load data ──────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Baseline AdaBoost ──────────────────────────────────────────────────────
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # decision stump
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME',
    random_state=42
)
ada.fit(X_train, y_train)

print(f"AdaBoost Test Accuracy: {ada.score(X_test, y_test):.4f}")

# ── 3. Staged predict — show accuracy per round ───────────────────────────────
train_errors, test_errors = [], []

for y_train_pred in ada.staged_predict(X_train):
    train_errors.append(1 - accuracy_score(y_train, y_train_pred))

for y_test_pred in ada.staged_predict(X_test):
    test_errors.append(1 - accuracy_score(y_test, y_test_pred))

best_round = np.argmin(test_errors) + 1
print(f"Best n_estimators (test): {best_round}, error: {test_errors[best_round-1]:.4f}")

# ── 4. Deeper base learner ────────────────────────────────────────────────────
ada_deep = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    algorithm='SAMME',
    random_state=42
)
ada_deep.fit(X_train, y_train)
print(f"AdaBoost (depth=3) Accuracy: {ada_deep.score(X_test, y_test):.4f}")

# ── 5. Estimator weights (α values) ──────────────────────────────────────────
print("\nFirst 10 estimator weights (α):")
for i, (weight, err) in enumerate(
    zip(ada.estimator_weights_[:10], ada.estimator_errors_[:10]), 1
):
    print(f"  Round {i:3d}: α = {weight:.4f},  ε = {err:.4f}")

# ── 6. Compare algorithms: SAMME vs SAMME.R ──────────────────────────────────
for algo in ['SAMME', 'SAMME.R']:
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200, learning_rate=1.0,
        algorithm=algo, random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Algorithm={algo:8s}: Test Accuracy = {clf.score(X_test, y_test):.4f}")
```

---

## Theoretical Guarantee: Training Error Bound

AdaBoost has an exponential training error bound:

```
Training error ≤ exp(-2 Σₜ γₜ²)

where γₜ = 0.5 - εₜ  (edge over random guessing)
```

If each weak learner has edge γ > 0 over random:
```
Training error ≤ exp(-2Tγ²) → 0 as T → ∞
```

This shows AdaBoost can achieve zero training error given sufficient rounds and learners that beat random chance.

---

## Interview Q&A

**Q1: What happens to the sample weights of correctly classified examples?**

> They decrease by a factor of `e^{-αₜ}`. As αₜ increases (better classifier), correctly classified samples get downweighted more aggressively, forcing the next learner to pay more attention to the harder examples.

**Q2: Why is AdaBoost sensitive to outliers/noisy labels?**

> Mislabelled samples are always misclassified, so their weights grow exponentially each round (`e^{+αₜ}` per round). Eventually, a single noisy sample can dominate the entire weight distribution, causing subsequent learners to memorise it. Solutions: robust boosting variants (BrownBoost, MadaBoost) or outlier removal preprocessing.

**Q3: What does the learning rate do in AdaBoost?**

> It shrinks each weak learner's contribution: `f(x) += learning_rate · αₜ · hₜ(x)`. Smaller learning rate requires more estimators but provides regularisation, often improving generalisation. There's a bias-variance tradeoff: lower learning rate + more estimators usually outperforms a single high learning rate.

**Q4: Can AdaBoost overfit?**

> Theoretically, with ideal weak learners, training error decreases exponentially but generalisation error can plateau or increase. In practice, AdaBoost is relatively resistant to overfitting compared to single trees, but it does overfit on noisy data when `n_estimators` is very large. Use staged predict + validation set to find the optimal stopping point.

**Q5: Why must each weak learner perform better than random chance (εₜ < 0.5)?**

> If εₜ ≥ 0.5, αₜ ≤ 0, meaning the learner's vote is zero or negative. The training error bound requires γₜ = 0.5 - εₜ > 0 for convergence. In practice, decision stumps (depth-1 trees) virtually always satisfy this on binary problems.

**Q6: How does AdaBoost relate to gradient boosting?**

> AdaBoost can be derived as gradient descent in function space minimising the **exponential loss** `L(y, f) = e^{-y·f(x)}`. Gradient boosting generalises this framework to arbitrary differentiable loss functions, replacing the analytical reweighting with numerical gradient computation. This connection was elucidated by Friedman, Hastie, and Tibshirani (2000).

---

## Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| Sensitive to outliers | Misclassified samples upweighted exponentially | Clean data; use robust loss functions |
| Slow convergence | High learning rate, stumps | Reduce `learning_rate`, increase `n_estimators` |
| Overfitting on noise | Outliers dominate weight distribution | Remove outliers; use `max_depth=2–3` |
| Doesn't support `predict_proba` well | SAMME (discrete) has no probabilities | Use SAMME.R or Platt scaling |
| Slow training | Sequential algorithm | Cannot parallelise (unlike Random Forest) |

---

## Quick Reference Cheat Sheet

```
ADABOOST — QUICK REFERENCE
═══════════════════════════════════════════════════════════

ALGORITHM
  1. Initialise: D₁(i) = 1/n
  2. For t = 1 to T:
     a. Train hₜ on distribution Dₜ
     b. εₜ = Σᵢ Dₜ(i) · 𝟙[hₜ(xᵢ) ≠ yᵢ]
     c. αₜ = ½ ln((1-εₜ)/εₜ)
     d. Dₜ₊₁(i) ∝ Dₜ(i) · exp(-αₜ yᵢ hₜ(xᵢ))
  3. H(x) = sign(Σₜ αₜ hₜ(x))

KEY FORMULAS
  Alpha       : αₜ = ½ ln((1-εₜ)/εₜ)
  Weight up   : ×e^{+αₜ}  (misclassified)
  Weight down : ×e^{-αₜ}  (correct)
  Train error ≤ exp(-2Σγₜ²)

KEY HYPERPARAMETERS
  n_estimators  → number of boosting rounds (tune via staged_predict)
  learning_rate → shrinkage per step (lower + more estimators = better)
  max_depth     → depth of base tree (1 = stump, default)
  algorithm     → 'SAMME' (discrete) or 'SAMME.R' (real, default)

TYPICAL SKLEARN USAGE
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.tree import DecisionTreeClassifier
  ada = AdaBoostClassifier(
      estimator=DecisionTreeClassifier(max_depth=1),
      n_estimators=200, learning_rate=0.5,
      algorithm='SAMME', random_state=42
  )

PROS                         CONS
  ✅ Simple, interpretable     ❌ Sensitive to noise/outliers
  ✅ Few hyperparameters       ❌ Sequential (not parallelisable)
  ✅ Strong theoretical bounds ❌ Slower than RF on same n_est
  ✅ Works with any weak clf   ❌ Struggles with multiclass
═══════════════════════════════════════════════════════════
```
