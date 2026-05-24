# Hyperparameter Tuning — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | The process of finding the model configuration (hyperparameters) that maximizes validation performance |
| **WHY** | Default hyperparameters are rarely optimal; tuning can significantly improve model performance |
| **WHEN** | After establishing a baseline; before final evaluation on test set |
| **HOW** | Grid Search (exhaustive), Random Search (efficient), Bayesian Optimization (intelligent sequential search) |
| **PITFALLS** | Tuning on test set (data leakage), overfitting to validation set, ignoring learning curves, wrong CV strategy |

---

## 1. Grid Search — Exhaustive Enumeration

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}

# Total combinations = 3 × 4 × 3 × 2 = 72
# With 5-fold CV: 72 × 5 = 360 model fits
print(f"Total fits: {3*4*3*2 * 5}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X, y)

print(f"\nBest params:  {grid_search.best_params_}")
print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")

# Inspect results
import pandas as pd
results_df = pd.DataFrame(grid_search.cv_results_)
results_df_sorted = results_df.sort_values("mean_test_score", ascending=False)
print("\nTop 5 configurations:")
print(results_df_sorted[["params", "mean_test_score", "std_test_score"]].head())
```

**Pros:** Exhaustive — guaranteed to find the best in the grid.
**Cons:** Exponentially expensive (curse of dimensionality in hyperparameter space).

---

## 2. Random Search — Efficient Sampling

> "Random search outperforms grid search because hyperparameters differ in their importance. Grid search wastes compute exploring many values of unimportant hyperparameters." — Bergstra & Bengio (2012)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

# Continuous distributions (Random Search advantage over Grid Search)
param_dist = {
    "n_estimators": randint(100, 500),          # Uniform integer [100, 500)
    "max_depth": [None] + list(range(3, 25)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.1, 0.9),          # Fraction of features [0.1, 1.0)
    "bootstrap": [True, False]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,          # Try 50 random configurations (vs 360+ for Grid Search)
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

random_search.fit(X, y)
print(f"Best params:  {random_search.best_params_}")
print(f"Best ROC-AUC: {random_search.best_score_:.4f}")
```

**Key distributions from scipy.stats:**
| Distribution | Use For | Example |
|--------------|---------|---------|
| `randint(a, b)` | Integer hyperparameters | `n_estimators`, `max_depth` |
| `uniform(a, scale)` | Float in [a, a+scale) | Learning rate (linear scale) |
| `loguniform(a, b)` | Float on log scale | `C` in SVM, `alpha` in Ridge |
| `expon(scale)` | Exponential distribution | Regularization strength |

---

## 3. Bayesian Optimization with Optuna

Bayesian Optimization builds a probabilistic surrogate model of the objective function and uses it to **intelligently choose** the next hyperparameter configuration to evaluate.

```python
import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def objective(trial):
    """Optuna objective function — define search space here"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    model = GradientBoostingClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

# Create study with Tree-structured Parzen Estimator (TPE) sampler
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest trial value: {study.best_trial.value:.4f}")
print(f"Best params:\n{study.best_trial.params}")

# Visualize optimization history (if plotly available)
try:
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()
except Exception:
    pass
```

**Optuna Advanced Features:**

```python
# Pruning — stop unpromising trials early
from optuna.integration import SklearnPruningCallback
from sklearn.neural_network import MLPClassifier

def objective_with_pruning(trial):
    hidden_layer_sizes = trial.suggest_categorical(
        "hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64), (256, 128)]
    )
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=200,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()

study_mlp = optuna.create_study(direction="maximize")
study_mlp.optimize(objective_with_pruning, n_trials=30)
print(f"Best MLP ROC-AUC: {study_mlp.best_value:.4f}")
```

---

## 4. Learning Curves — Diagnosing Under/Overfitting

Learning curves reveal whether adding more data or tuning model complexity will help.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def plot_learning_curve(estimator, title, X, y, cv=5, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="roc_auc", n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="blue",   label="Train AUC")
    ax.plot(train_sizes, val_mean,   "s-", color="orange", label="Validation AUC")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std,     val_mean + val_std,     alpha=0.1, color="orange")
    ax.set_title(title)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    ax.grid(True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Underfitting model
underfitting_model = GradientBoostingClassifier(n_estimators=10, max_depth=1, random_state=42)
plot_learning_curve(underfitting_model, "Underfitting (n_estimators=10, depth=1)", X, y, ax=axes[0])

# Overfitting model
overfitting_model = GradientBoostingClassifier(n_estimators=200, max_depth=8,
                                               min_samples_leaf=1, random_state=42)
plot_learning_curve(overfitting_model, "Overfitting (n_estimators=200, depth=8)", X, y, ax=axes[1])

plt.tight_layout()
plt.show()
```

**Reading Learning Curves:**
```
UNDERFITTING (High Bias):
  Train AUC: Low and flat
  Val AUC:   Low
  Gap:       Small
  Fix: More complex model, fewer constraints

OVERFITTING (High Variance):
  Train AUC: High
  Val AUC:   Much lower
  Gap:       Large
  Fix: Regularize, more data, simpler model

GOOD FIT:
  Train AUC: High
  Val AUC:   Close to Train AUC
  Gap:       Small
```

---

## 5. Typical Hyperparameter Ranges

### Random Forest

```python
rf_param_grid = {
    "n_estimators": [100, 200, 500],            # More trees = more stable, diminishing returns
    "max_depth": [None, 5, 10, 15, 20],         # None = full depth (may overfit)
    "min_samples_split": [2, 5, 10],            # Larger = more regularization
    "min_samples_leaf": [1, 2, 4],              # Larger = smoother boundaries
    "max_features": ["sqrt", "log2", 0.5],      # sqrt is default and often good
    "bootstrap": [True, False],
    "max_samples": [0.8, 0.9, None]             # Bagging fraction
}
```

### XGBoost

```python
xgb_param_dist = {
    "n_estimators": randint(100, 1000),
    "max_depth": randint(3, 10),                # Usually 3-8
    "learning_rate": loguniform(0.01, 0.3),     # AKA eta; lower = better with more trees
    "subsample": uniform(0.5, 0.5),             # Row sampling [0.5, 1.0]
    "colsample_bytree": uniform(0.5, 0.5),      # Column sampling [0.5, 1.0]
    "min_child_weight": randint(1, 10),         # Regularization
    "gamma": loguniform(1e-8, 1.0),             # Min loss reduction for split
    "reg_alpha": loguniform(1e-8, 1.0),         # L1 regularization
    "reg_lambda": loguniform(1e-8, 10.0),       # L2 regularization
}
```

### LightGBM

```python
lgb_param_dist = {
    "n_estimators": randint(100, 1000),
    "max_depth": [-1] + list(range(3, 12)),     # -1 = no limit
    "num_leaves": randint(20, 300),             # < 2^max_depth to prevent overfit
    "learning_rate": loguniform(0.01, 0.3),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_samples": randint(5, 100),       # Min samples per leaf
    "reg_alpha": loguniform(1e-8, 10.0),
    "reg_lambda": loguniform(1e-8, 10.0),
}
```

**Key relationship:** `num_leaves < 2^max_depth` in LightGBM. Setting `num_leaves=31` with `max_depth=-1` is the default (leaf-wise growth).

---

## 6. Halving Search — Efficient with Many Candidates

```python
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# HalvingRandomSearchCV: Start with many candidates on small data,
# promote only top performers to next round with more data
halving_search = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions={
        "n_estimators": randint(50, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
    },
    n_candidates=100,   # Start with 100 configurations
    factor=3,           # Keep top 1/3 each round
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

halving_search.fit(X, y)
print(f"Best params:  {halving_search.best_params_}")
print(f"Best AUC:     {halving_search.best_score_:.4f}")
```

---

## 7. Interview Q&A

**Q1: Why is Random Search often better than Grid Search?**
> Bergstra & Bengio (2012) showed that in practice, most hyperparameter combinations have similar performance because only a few hyperparameters (typically 1-2) are truly important. Grid Search wastes compute evaluating many combinations that vary only in unimportant dimensions. Random Search covers a wider range of the truly important hyperparameters with fewer total evaluations.

**Q2: Explain Bayesian Optimization. How is it different from Random/Grid Search?**
> Bayesian Optimization maintains a probabilistic surrogate model (typically Gaussian Process or TPE) of the objective function. After each evaluation, it updates the surrogate and uses an acquisition function (e.g., Expected Improvement) to select the next point that balances exploration (trying unknown regions) and exploitation (refining known good regions). Unlike Random/Grid Search, it uses information from previous trials to make informed decisions, converging faster especially with expensive-to-evaluate objectives.

**Q3: How do you avoid overfitting to the validation set during hyperparameter tuning?**
> (1) Use cross-validation instead of a single validation split. (2) Use nested CV when you want an unbiased final estimate. (3) Keep a held-out test set that is never touched during tuning. (4) Limit the number of hyperparameter evaluations. (5) Use early stopping on validation performance.

**Q4: What is the learning rate vs n_estimators tradeoff in gradient boosting?**
> Lower learning rate requires more estimators to reach the same performance, but typically achieves better generalization. The optimal strategy: set learning_rate to a small value (0.01–0.05) and use early stopping to determine the optimal n_estimators. This avoids overfitting while achieving good performance.

**Q5: How do you tune an XGBoost model in practice?**
> Recommended order: (1) Fix a moderate learning_rate (0.1) and n_estimators (use early stopping). (2) Tune tree structure: max_depth, min_child_weight. (3) Tune sampling: subsample, colsample_bytree. (4) Tune regularization: gamma, reg_alpha, reg_lambda. (5) Finally, lower learning_rate and increase n_estimators for fine-tuning.

**Q6: What does a learning curve tell you about your model?**
> It reveals the bias-variance tradeoff in practice. If training and validation error both converge at a high error level (small gap), the model is underfitting — increase complexity. If training error is low but validation error is high (large gap), the model is overfitting — add regularization or collect more data. If both are low and converging, the model is well-tuned.

---

## 8. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Test set leakage** | Tuning hyperparameters using test set | Use separate validation set or nested CV |
| **Too coarse grid** | Grid misses optimal region | Start coarse, refine around best params |
| **Ignoring correlations** | Grid Search treats params independently | Use Random/Bayesian to explore joint space |
| **No early stopping** | Training boosting too long | Use eval set + early_stopping_rounds |
| **Default threshold** | Not tuning decision threshold after model | Tune threshold on validation set |
| **Over-tuning** | Too many trials → val set overfitting | Limit n_trials; use nested CV for estimate |
| **Wrong CV for time series** | Using StratifiedKFold on time series | Use TimeSeriesSplit |

---

## 9. Quick Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│               HYPERPARAMETER TUNING CHEAT SHEET                      │
├──────────────────┬──────────────┬──────────────┬────────────────────┤
│ Method           │ Speed        │ Quality      │ Use When           │
├──────────────────┼──────────────┼──────────────┼────────────────────┤
│ Grid Search      │ Slow         │ Thorough     │ Few params, small  │
│                  │              │              │ grid               │
│ Random Search    │ Fast         │ Good         │ Many params, large │
│                  │              │              │ space              │
│ Bayesian (Optuna)│ Moderate     │ Best         │ Expensive to train,│
│                  │              │              │ many trials needed │
│ Halving Search   │ Fast         │ Good         │ Many candidates,   │
│                  │              │              │ progressive budget  │
├──────────────────┴──────────────┴──────────────┴────────────────────┤
│ RULE: Never tune on test set! Use validation set or nested CV        │
│ RULE: More trees → lower variance, diminishing returns               │
│ Learning rate ↓ + n_estimators ↑ = better generalization            │
│ num_leaves < 2^max_depth (LightGBM)                                  │
└──────────────────────────────────────────────────────────────────────┘
```
