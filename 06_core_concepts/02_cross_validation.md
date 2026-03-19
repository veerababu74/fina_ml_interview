# Cross-Validation — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | A resampling technique to estimate model generalization performance using the available data |
| **WHY** | A single train/test split has high variance in the performance estimate; CV averages over multiple splits |
| **WHEN** | Always — especially with limited data. Use stratified CV for classification, TimeSeriesSplit for temporal data |
| **HOW** | Split data into k folds; train on k-1, evaluate on the held-out fold; repeat k times and average |
| **PITFALLS** | Data leakage (preprocessing outside CV), wrong CV for time series (no random shuffle!), nested CV for hyperparameter tuning |

---

## 1. Why Cross-Validation?

A single 80/20 split has **high variance** — you may get lucky or unlucky with which samples end up in the test set. Cross-validation reduces this variance by averaging over multiple splits.

```
Single split:
 [====TRAIN====|==TEST==]   ← One estimate, high variance

5-Fold CV:
 [-----|=TEST=|-----|-----|-----]  Fold 1
 [=====|-----|=TEST=|-----|-----]  Fold 2
 [=====|-----|-----|=TEST=|-----]  Fold 3
 [=====|-----|-----|-----|=TEST=]  Fold 4
 [=TEST=|-----|-----|-----|-----]  Fold 5

Final score = mean ± std of 5 fold scores  ← Lower variance
```

---

## 2. k-Fold Cross-Validation

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, cross_validate

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)

# Basic k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

print(f"CV Scores: {scores}")
print(f"Mean:  {scores.mean():.4f}")
print(f"Std:   {scores.std():.4f}")
print(f"95% CI: ({scores.mean() - 2*scores.std():.4f}, "
      f"{scores.mean() + 2*scores.std():.4f})")

# Multiple metrics at once
cv_results = cross_validate(
    model, X, y, cv=kf,
    scoring=["accuracy", "f1", "roc_auc"],
    return_train_score=True
)
for metric in ["test_accuracy", "test_f1", "test_roc_auc"]:
    vals = cv_results[metric]
    print(f"{metric}: {vals.mean():.4f} ± {vals.std():.4f}")
```

**Choosing k:**
- **k=5:** Standard default — good bias-variance balance in the CV estimate
- **k=10:** More folds → lower bias in estimate, higher variance, more compute
- **k=n (LOOCV):** Leave-One-Out — unbiased but high variance and expensive; useful for very small datasets

---

## 3. Stratified k-Fold (Classification)

Standard k-Fold doesn't preserve class ratios. With imbalanced data, some folds may have no positive class examples.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Check class ratios in each fold
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    y_val_fold = y[val_idx]
    pos_ratio = y_val_fold.mean()
    print(f"Fold {fold_idx + 1}: n={len(val_idx)}, positive_ratio={pos_ratio:.3f}")

# StratifiedKFold ensures each fold maintains the same class ratio as the full dataset
scores_stratified = cross_val_score(
    model, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1"
)
print(f"\nStratified CV F1: {scores_stratified.mean():.4f} ± {scores_stratified.std():.4f}")
```

**Always use StratifiedKFold for classification tasks.**

---

## 4. TimeSeriesSplit — NEVER Shuffle Temporal Data!

For time series, future data **must not** be used to train on past predictions. Standard k-Fold randomly shuffles, causing data leakage.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# TimeSeriesSplit maintains temporal order — train always comes before test
tscv = TimeSeriesSplit(n_splits=5, gap=0)

# Visualize the splits
n_samples = 100
X_ts = np.arange(n_samples).reshape(-1, 1)

fig, axes = plt.subplots(5, 1, figsize=(12, 8))
for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    axes[fold_idx].scatter(train_idx, np.ones_like(train_idx),
                           c="steelblue", label="Train", marker="|", s=100)
    axes[fold_idx].scatter(test_idx, np.ones_like(test_idx),
                           c="orange", label="Test", marker="|", s=100)
    axes[fold_idx].set_title(
        f"Fold {fold_idx+1}: Train [0-{train_idx[-1]}], Test [{test_idx[0]}-{test_idx[-1]}]"
    )
    axes[fold_idx].set_yticks([])
plt.tight_layout()
plt.show()

# Cross-validate a time series model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Simulated time series data
np.random.seed(42)
t = np.arange(200)
y_ts = np.sin(2 * np.pi * t / 20) + 0.3 * np.random.randn(200)
X_ts_features = np.column_stack([
    t,
    np.sin(2 * np.pi * t / 20),
    np.cos(2 * np.pi * t / 20)
])

tscv = TimeSeriesSplit(n_splits=5)
model_ts = GradientBoostingRegressor(n_estimators=100, random_state=42)
mae_scores = []

for train_idx, test_idx in tscv.split(X_ts_features):
    X_tr, X_te = X_ts_features[train_idx], X_ts_features[test_idx]
    y_tr, y_te = y_ts[train_idx], y_ts[test_idx]
    model_ts.fit(X_tr, y_tr)
    y_pred = model_ts.predict(X_te)
    mae_scores.append(mean_absolute_error(y_te, y_pred))

print(f"TimeSeriesCV MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
```

**TimeSeriesSplit Rules:**
1. **Never shuffle** temporal data before splitting
2. Use `gap` parameter to simulate prediction lag (e.g., predict 1 week ahead)
3. Training set always precedes test set in time

---

## 5. Nested Cross-Validation for Hyperparameter Tuning

**Problem:** If you use the same CV splits for both hyperparameter tuning and final evaluation, you get an optimistic (biased) estimate of generalization performance.

**Solution:** Nested CV — outer loop evaluates generalization, inner loop selects hyperparameters.

```python
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=300, n_features=10, random_state=42)

# WRONG WAY: Tune hyperparameters, then evaluate with same CV → optimistic bias
param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

svc = SVC(kernel="rbf", random_state=42)

# CORRECT WAY: Nested CV
grid_search = GridSearchCV(svc, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring="accuracy")

print("=== Nested CV ===")
print(f"Outer CV scores: {nested_scores.round(4)}")
print(f"Mean: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print("This is an unbiased estimate of generalization performance")

# Non-nested (optimistic) estimate
grid_search.fit(X, y)
non_nested_score = grid_search.best_score_
print(f"\nNon-nested (optimistic) score: {non_nested_score:.4f}")
print(f"Optimism bias: {non_nested_score - nested_scores.mean():.4f}")
```

**When to use Nested CV:**
- When dataset is small and you want an unbiased performance estimate
- When reporting results in papers/competitions
- When model selection and evaluation use the same data

**When NOT to use:**
- Very large datasets (computational cost is O(n_outer × n_inner × n_configs))
- When you have enough data for separate train/validation/test splits

---

## 6. Data Leakage in Cross-Validation

**The most dangerous mistake:** Fitting preprocessors (scalers, encoders, imputers) on the full dataset before CV. The validation folds "see" the test distribution.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=200, n_features=20, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)

# ❌ WRONG WAY — Leakage: scaler sees all data before CV splits
scaler = StandardScaler()
X_scaled_wrong = scaler.fit_transform(X)  # Leaks test statistics into training!
score_wrong = cross_val_score(model, X_scaled_wrong, y, cv=cv, scoring="accuracy")
print(f"❌ With leakage:    {score_wrong.mean():.4f} ± {score_wrong.std():.4f}")

# ✅ CORRECT WAY — Pipeline ensures scaler is fit only on train folds
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
score_correct = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"✅ No leakage:      {score_correct.mean():.4f} ± {score_correct.std():.4f}")
```

**Other common leakage sources:**
- Fitting imputers on full data before CV
- Computing feature statistics (mean, std, target encoding) on full data
- Selecting features using the full dataset before CV
- Time series: using future data to compute lag features

---

## 7. Pipeline to Prevent Leakage — Complete Example

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

# Mixed-type dataset with missing values
df = pd.DataFrame({
    "age":      np.random.randint(20, 70, n).astype(float),
    "income":   np.random.exponential(50000, n),
    "category": np.random.choice(["A", "B", "C"], n),
    "label":    np.random.randint(0, 2, n)
})
df.loc[np.random.choice(n, 20, replace=False), "age"] = np.nan
df.loc[np.random.choice(n, 15, replace=False), "income"] = np.nan

X = df.drop("label", axis=1)
y = df["label"].values

numeric_features = ["age", "income"]
categorical_features = ["category"]

# Define preprocessing per column type
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full pipeline: all preprocessing inside CV loop
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(
    full_pipeline, X, y, cv=cv,
    scoring=["accuracy", "f1", "roc_auc"],
    return_train_score=True
)

for metric in ["test_accuracy", "test_f1", "test_roc_auc"]:
    vals = results[metric]
    print(f"{metric}: {vals.mean():.4f} ± {vals.std():.4f}")
```

---

## 8. Interview Q&A

**Q1: Why do we use cross-validation instead of a single train/test split?**
> A single split gives one performance estimate with high variance — you might get an unusually easy or hard test set. CV averages over k different splits, producing a lower-variance estimate with confidence interval. It also uses the data more efficiently: every sample appears in a test fold exactly once.

**Q2: What's the difference between k-Fold and Stratified k-Fold?**
> Standard k-Fold splits data randomly without regard for class distribution. If classes are imbalanced, some folds might have very few (or zero) positive examples, making the metric unreliable. Stratified k-Fold ensures each fold preserves the original class ratio. Always use Stratified k-Fold for classification, especially with imbalanced data.

**Q3: Why can't we shuffle time series data before CV?**
> Shuffling breaks the temporal dependence structure and causes data leakage. If we shuffle, a training fold may contain data from the future (relative to the test fold), giving the model information it wouldn't have in production. TimeSeriesSplit ensures training data always precedes test data in time, mimicking the real deployment scenario.

**Q4: What is nested CV and when is it needed?**
> Nested CV uses an outer loop to evaluate generalization and an inner loop to select hyperparameters. Without nesting, the CV that selects hyperparameters is the same one that estimates performance — optimizing over it inflates the performance estimate. Nested CV gives an unbiased estimate: the outer loop never sees hyperparameter selection choices.

**Q5: Give an example of data leakage in CV.**
> Fitting a StandardScaler on the full dataset before CV is classic leakage. The scaler computes mean/std using test fold data, which then influences how training folds are normalized. The model has "seen" the test distribution. Fix: wrap everything in a Pipeline so the scaler is re-fit on each training fold independently.

**Q6: How do you choose the number of folds k?**
> Rule of thumb: k=5 or k=10 works for most cases. More folds (larger k) → lower bias in the CV estimate (each fold is trained on more data) but higher variance and more compute. k=n (LOOCV) is unbiased but has high variance and is expensive. With small datasets (<500 samples), k=10 or LOOCV is preferred. With large datasets, k=3 or k=5 is sufficient.

---

## 9. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Preprocessing leakage** | Fit scaler/imputer on full data before CV | Use Pipeline — fit inside CV loop |
| **Shuffling time series** | Using KFold on temporal data | Use TimeSeriesSplit |
| **Non-stratified classification** | KFold on imbalanced classes | Use StratifiedKFold |
| **Biased tuning estimate** | Using same CV for tuning and evaluation | Use nested CV or separate test set |
| **Ignoring std** | Reporting only mean CV score | Report mean ± std; high std = unstable |
| **LOOCV variance** | High variance in LOOCV estimate | Use k=5/10 instead for most cases |
| **Group leakage** | Same patient/user in train and test | Use GroupKFold to keep groups intact |

---

## 10. Quick Reference Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CROSS-VALIDATION CHEAT SHEET                      │
├───────────────────┬───────────────────────┬─────────────────────────┤
│ CV Type           │ Use When              │ Key Parameter           │
├───────────────────┼───────────────────────┼─────────────────────────┤
│ KFold             │ Regression            │ n_splits, shuffle=True  │
│ StratifiedKFold   │ Classification        │ n_splits, shuffle=True  │
│ TimeSeriesSplit   │ Time series           │ n_splits, gap           │
│ GroupKFold        │ Repeated subjects     │ groups=subject_id       │
│ LOOCV             │ Tiny datasets (<100)  │ LeaveOneOut()           │
│ Nested CV         │ Hyperparameter tuning │ inner + outer CV        │
├───────────────────┴───────────────────────┴─────────────────────────┤
│ RULE: Always use Pipeline to prevent preprocessing leakage!          │
│ RULE: Never shuffle time series data!                                │
│ RULE: Use StratifiedKFold for all classification tasks!              │
│ k=5 or k=10 is the default recommendation for general use           │
└─────────────────────────────────────────────────────────────────────┘
```
