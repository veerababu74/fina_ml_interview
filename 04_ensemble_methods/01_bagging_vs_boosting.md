# Bagging vs Boosting

## 1. Core Concept

Both bagging and boosting combine multiple **weak learners** into a strong ensemble. They differ fundamentally in *how* the weak learners are trained and *what error they target*.

| | Bagging | Boosting |
|--|---------|---------|
| **Training order** | Parallel (independent) | Sequential (dependent) |
| **Error targeted** | Reduces **variance** | Reduces **bias** |
| **Sample weights** | Equal (bootstrap resample) | Re-weighted by errors |
| **Learner strength** | Low-bias, high-variance (deep trees) | High-bias, low-variance (stumps) |
| **Overfitting risk** | Low | Higher (early stopping needed) |
| **Speed** | Fast (parallelisable) | Slower (sequential) |
| **Examples** | Random Forest, Extra Trees | AdaBoost, Gradient Boosting, XGBoost |

---

## 2. Bagging — Parallel Training

### 2.1 Algorithm

```
Given: training set D = {(x₁,y₁), …, (xₙ,yₙ)}, number of models T

For t = 1 to T:
  1. Draw bootstrap sample Dₜ by sampling n points from D WITH replacement.
     ~63.2% of unique points appear in each Dₜ; ~36.8% are out-of-bag (OOB).
  2. Train base learner hₜ on Dₜ independently.

Prediction (regression):   ĥ(x) = (1/T) Σₜ hₜ(x)
Prediction (classification): ĥ(x) = majority_vote{hₜ(x)}
```

### 2.2 Why Bagging Reduces Variance — Mathematical Proof

Let each base learner hₜ have variance σ² and pairwise correlation ρ (due to using same training data).

```
Var(ĥ) = Var((1/T) Σₜ hₜ)
        = (1/T²) [T·σ² + T(T-1)·ρ·σ²]
        = σ²/T + ρσ²(T-1)/T
        = ρσ² + σ²(1-ρ)/T

As T → ∞:
  Var(ĥ) → ρσ²
```

**Key insights:**
- If base learners are **uncorrelated** (ρ=0): Var(ĥ) → 0 as T → ∞ (perfect variance reduction).
- If ρ=1 (perfectly correlated): Var(ĥ) = σ² (no improvement).
- Bootstrap resampling creates diversity (different Dₜ) → lower ρ.
- Random Forest reduces ρ further by also randomising feature subsets at each split.

### 2.3 Out-of-Bag (OOB) Evaluation

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',       # feature randomness (key for diversity)
    max_depth=None,
    oob_score=True,            # use OOB samples for free validation
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)

print(f"OOB Score:        {rf.oob_score_:.4f}")
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# OOB score ≈ CV score — essentially free cross-validation!
```

---

## 3. Boosting — Sequential Training

### 3.1 Algorithm (AdaBoost)

```
1. Initialise uniform sample weights: wᵢ = 1/n for all i.
2. For t = 1 to T:
   a. Train weak learner hₜ on weighted dataset.
   b. Compute weighted error:
        εₜ = Σᵢ wᵢ · 1[hₜ(xᵢ) ≠ yᵢ]  /  Σᵢ wᵢ
   c. Compute learner weight:
        αₜ = 0.5 · ln((1 − εₜ) / εₜ)      [high weight if low error]
   d. Update sample weights:
        wᵢ ← wᵢ · exp(−αₜ · yᵢ · hₜ(xᵢ))
        Normalise: wᵢ ← wᵢ / Σⱼ wⱼ
        [Misclassified points get higher weight, correctly classified get lower]

Final prediction: H(x) = sign(Σₜ αₜ · hₜ(x))
```

### 3.2 Why Boosting Reduces Bias

Each subsequent learner focuses on **mistakes** of the ensemble so far. The ensemble gradually corrects systematic errors (bias), fitting the residuals rather than the original targets.

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# AdaBoost: shallow trees (stumps)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=1.0,
    algorithm='SAMME',
    random_state=42
)
ada.fit(X_train, y_train)
print(f"AdaBoost Accuracy:  {accuracy_score(y_test, ada.predict(X_test)):.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,           # stochastic GB — adds bagging-like variance reduction
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Gradient Boost Acc: {accuracy_score(y_test, gb.predict(X_test)):.4f}")
```

---

## 4. Restaurant Analogy

> **Bagging = asking 200 independent chefs to cook the same dish.**
> Each chef uses a slightly different set of ingredients (bootstrap sample). The final dish is the average of all 200 plates. Even if one chef has an off day, the others compensate. The final result is smooth, consistent, and rarely terrible — but each individual chef is already fairly skilled.

> **Boosting = training one apprentice chef, then hiring another apprentice who specialises in the dishes the first one got wrong, then another who fixes the second's mistakes, and so on.**
> Each new apprentice focuses on correcting the mistakes of the previous ensemble. The team gradually improves at the hardest dishes. But if you keep training apprentices for too long (too many rounds), they start memorising individual orders — overfitting.

---

## 5. Gradient Boosting — Residual Fitting Intuition

Gradient Boosting frames boosting as **gradient descent in function space**:

```
At step t, the ensemble is F_{t}(x) = F_{t-1}(x) + η·hₜ(x)

hₜ is trained on the **pseudo-residuals** (negative gradient of loss):
  rᵢₜ = −∂L(yᵢ, F_{t-1}(xᵢ)) / ∂F_{t-1}(xᵢ)

For MSE loss: rᵢₜ = yᵢ − F_{t-1}(xᵢ)  (true residuals)
For log-loss: rᵢₜ = yᵢ − p̂ᵢ            (residuals in probability space)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Track training and test error vs number of trees
gb = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)

train_errors = [mean_squared_error(y_train, pred) for pred in gb.staged_predict(X_train)]
test_errors  = [mean_squared_error(y_test,  pred) for pred in gb.staged_predict(X_test)]

best_n = np.argmin(test_errors) + 1
print(f"Best n_estimators: {best_n}  (test RMSE = {np.sqrt(test_errors[best_n-1]):.2f})")

plt.figure(figsize=(9, 4))
plt.plot(np.sqrt(train_errors), label='Train RMSE')
plt.plot(np.sqrt(test_errors),  label='Test RMSE')
plt.axvline(x=best_n, color='red', linestyle='--', label=f'Best n={best_n}')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.title('Gradient Boosting: Staged Error')
plt.legend()
plt.tight_layout()
plt.savefig('gb_staged_error.png', dpi=100)
plt.show()
```

---

## 6. Performance Comparison

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import time

X, y = make_classification(n_samples=2000, n_features=20, n_informative=12,
                            n_redundant=4, random_state=42)

models = {
    'Single Decision Tree':     DecisionTreeClassifier(random_state=42),
    'Random Forest (Bagging)':  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'Extra Trees':              ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'AdaBoost':                 AdaBoostClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':        GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                           max_depth=3, random_state=42),
}

results = []
for name, model in models.items():
    start = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    elapsed = time.time() - start
    results.append({'Model': name, 'Accuracy': scores.mean(),
                    'Std': scores.std(), 'Time (s)': round(elapsed, 2)})

df = pd.DataFrame(results).set_index('Model')
df['Accuracy'] = df['Accuracy'].map('{:.4f}'.format)
df['Std']      = df['Std'].map('{:.4f}'.format)
print(df.to_string())
```

**Typical results:**
| Method | Accuracy | Notes |
|--------|---------|-------|
| Single Tree | ~0.87 | Baseline |
| Random Forest | ~0.93 | Low variance, robust |
| Extra Trees | ~0.92 | Faster, more random splits |
| AdaBoost | ~0.91 | Sensitive to noise |
| Gradient Boosting | ~0.94 | Best accuracy, slowest |

---

## 7. XGBoost vs LightGBM vs CatBoost

```python
# Performance comparison of modern boosting libraries
# pip install xgboost lightgbm catboost

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(n_samples=5000, n_features=30, n_informative=15, random_state=42)

try:
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=6, use_label_encoder=False,
                                    eval_metric='logloss', random_state=42, verbosity=0)
    s = cross_val_score(xgb_model, X, y, cv=5).mean()
    print(f"XGBoost CV Accuracy:   {s:.4f}")
except ImportError:
    print("XGBoost not installed: pip install xgboost")

try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                     num_leaves=63, random_state=42, verbose=-1)
    s = cross_val_score(lgb_model, X, y, cv=5).mean()
    print(f"LightGBM CV Accuracy:  {s:.4f}")
except ImportError:
    print("LightGBM not installed: pip install lightgbm")
```

---

## 8. Interview Q&A

**Q1: Why does bagging reduce variance but not bias?**  
A: Bagging averages T models trained on bootstrap samples. Averaging T unbiased estimators gives an unbiased estimator (bias stays the same). But variance decreases: Var(average) = σ²/T + ρσ²(T−1)/T. Since we don't change the model architecture, the base learner's bias is unchanged. Bagging works best when base learners already have low bias (deep trees).

**Q2: Why does boosting reduce bias?**  
A: Boosting trains each new model on the residuals (mistakes) of the current ensemble. This is gradient descent in function space — each step moves the prediction closer to the true target by correcting the current prediction's systematic errors. This directly reduces bias. However, with too many rounds, boosting can overfit (increase variance).

**Q3: What is the key difference between AdaBoost and Gradient Boosting?**  
A: AdaBoost re-weights misclassified training points and trains the next learner on this re-weighted dataset; learner weights αₜ depend on error rate. Gradient Boosting fits the next learner to pseudo-residuals (negative gradient of the loss function), making it more general and applicable to any differentiable loss. Gradient Boosting is more flexible and usually outperforms AdaBoost.

**Q4: Can bagging overfit?**  
A: Much less than single models. Because individual learners are averaged, the variance of the ensemble is lower. However, if all base learners are highly correlated (e.g., all trained on very similar bootstraps), overfitting can still occur. Random Forest mitigates this with random feature selection. Bagging does not prevent high-bias models from underfitting.

**Q5: What is the Out-of-Bag (OOB) score?**  
A: Each bootstrap sample leaves out ~36.8% of training points (OOB samples). These OOB points can be used as a validation set for the trees that didn't train on them — providing a free cross-validation estimate. `oob_score=True` in sklearn enables this. OOB score closely approximates 5-fold CV without the computational cost.

**Q6: When would you prefer boosting over bagging?**  
A: Boosting when you need maximum predictive accuracy and your base learners are weak (high bias). Bagging when you need stability, robustness to noise/outliers, and parallelisable training. In practice, gradient boosting (XGBoost/LightGBM) often wins on tabular data competitions, but Random Forest is more robust out-of-the-box.

---

## 9. Common Pitfalls

1. **Applying boosting to noisy data** — AdaBoost assigns high weights to noisy/mislabelled points, causing overfitting; use bagging instead.
2. **Forgetting early stopping in boosting** — staged_predict or `validation_fraction` + `n_iter_no_change` prevent overfitting.
3. **Too high learning rate in boosting** — use 0.01–0.1 and compensate with more trees; `learning_rate × n_estimators` ≈ constant.
4. **Using deep trees in boosting** — boosting benefits from high-bias (shallow) learners; `max_depth=3–6` is typical.
5. **Ignoring feature importance correlation** — impurity-based importance overestimates high-cardinality features; use permutation importance.
6. **Not using `n_jobs=-1` for Random Forest** — bagging is perfectly parallel; always parallelise.
7. **Assuming OOB score replaces proper test set** — OOB is a training set estimate; always hold out a true test set for final evaluation.

---

## 10. Quick Reference Cheat Sheet

```
BAGGING:
  Training:    Parallel — bootstrap resample, train independently
  Error:       Reduces variance (averaging T learners)
  Math:        Var(avg) = ρσ² + σ²(1−ρ)/T → ρσ² as T→∞
  Base learner: Deep trees (low bias, high variance)
  OOB:         Free CV estimate (oob_score=True)
  Example:     RandomForest(n_estimators=200, max_features='sqrt', n_jobs=-1)

BOOSTING:
  Training:    Sequential — each model fits residuals of previous
  Error:       Reduces bias (gradient descent in function space)
  Weak learner: Shallow trees / stumps (high bias, low variance)
  Overfitting: Possible — use early stopping, learning_rate < 0.1
  Example:     GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=3, subsample=0.8)

DECISION GUIDE:
  Noisy data / outliers   → Bagging (robust)
  Max accuracy            → Boosting (XGBoost/LightGBM)
  Speed / parallelism     → Bagging (Random Forest)
  High-bias base model    → Boosting fixes it
  High-variance base model → Bagging fixes it
```
