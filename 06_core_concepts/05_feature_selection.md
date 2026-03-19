# Feature Selection — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | The process of selecting the most relevant features from the dataset to improve model performance and interpretability |
| **WHY** | Reduces overfitting, speeds up training, removes noise, improves interpretability, handles the curse of dimensionality |
| **WHEN** | When p >> n, when you have highly correlated features, when model is overfitting, when latency/inference speed matters |
| **HOW** | Filter (statistics), Wrapper (model-based search), Embedded (regularization, importance), SHAP |
| **PITFALLS** | Selecting features on full data before CV (leakage), confusing correlation with causation, ignoring feature interactions |

---

## 1. Filter Methods — Statistics-Based Selection

Filter methods select features based on statistical properties, **independent of any model**.

### Correlation Filter

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=15, noise=20, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# 1. Correlation with target
target_corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
print("Correlation with target (top 10):")
print(target_corr.head(10))

# 2. Correlation heatmap — identify redundant features
plt.figure(figsize=(12, 10))
sns.heatmap(df[feature_names].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 3. Remove highly correlated features (pairwise)
def remove_correlated_features(df, threshold=0.9):
    """Remove one feature from each highly correlated pair."""
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]
    print(f"Dropping {len(to_drop)} correlated features: {to_drop}")
    return df.drop(columns=to_drop)

df_reduced = remove_correlated_features(df[feature_names], threshold=0.9)
print(f"\nFeatures: {len(feature_names)} → {df_reduced.shape[1]}")
```

---

### Mutual Information

Mutual information (MI) measures **any** statistical dependence — not just linear correlation. Better for detecting nonlinear relationships.

```python
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2
)
from sklearn.datasets import make_classification

X_cls, y_cls = make_classification(n_samples=500, n_features=20,
                                    n_informative=5, random_state=42)

# Mutual information (works with nonlinear relationships)
mi_scores = mutual_info_classif(X_cls, y_cls, random_state=42)
mi_df = pd.DataFrame({
    "feature": [f"f_{i}" for i in range(20)],
    "MI_score": mi_scores
}).sort_values("MI_score", ascending=False)

print("Top 8 features by Mutual Information:")
print(mi_df.head(8))

# SelectKBest with MI
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X_cls, y_cls)
selected_mask = selector.get_support()
print(f"\nSelected {selected_mask.sum()} features")
```

---

### Chi-Square Test (Classification with non-negative features)

```python
from sklearn.preprocessing import MinMaxScaler

# Chi-square requires non-negative features
X_cls_pos = MinMaxScaler().fit_transform(X_cls)  # Scale to [0, 1]
chi2_scores, p_values = chi2(X_cls_pos, y_cls)

chi2_df = pd.DataFrame({
    "feature":  [f"f_{i}" for i in range(20)],
    "chi2":     chi2_scores,
    "p_value":  p_values
}).sort_values("chi2", ascending=False)

# Select features where p-value < 0.05
significant = chi2_df[chi2_df["p_value"] < 0.05]
print(f"Statistically significant features: {len(significant)}")
print(significant[["feature", "chi2", "p_value"]].head(8))
```

**Filter methods comparison:**

| Method | Relationship | Target Type | Non-negative? |
|--------|-------------|-------------|---------------|
| Pearson/Spearman correlation | Linear/Monotonic | Continuous | No |
| Mutual Information | Any (nonlinear OK) | Both | No |
| Chi-Square | Statistical dependence | Categorical | YES (non-neg) |
| ANOVA F-test (`f_classif`) | Linear | Classification | No |
| F-regression | Linear | Regression | No |

---

## 2. Wrapper Methods — Model-Based Search

Wrapper methods use a model to score feature subsets. More powerful than filter but more expensive.

### RFE — Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# RFE: eliminate features one-by-one based on model importance
rfe = RFE(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    n_features_to_select=8,  # Keep 8 features
    step=1,                  # Remove 1 at a time (or fraction)
    verbose=1
)
rfe.fit(X_cls, y_cls)
selected_features_rfe = np.where(rfe.support_)[0]
print(f"RFE selected features (indices): {selected_features_rfe}")
print(f"Feature rankings (1 = selected):\n{rfe.ranking_}")
```

### RFECV — Automatic k Selection via Cross-Validation

```python
# RFECV: automatically finds optimal number of features using CV
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    step=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    min_features_to_select=1,
    n_jobs=-1
)
rfecv.fit(X_cls, y_cls)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {np.where(rfecv.support_)[0]}")

# Plot CV score vs number of features
plt.figure(figsize=(10, 5))
n_features_range = range(rfecv.min_features_to_select, len(rfecv.cv_results_["mean_test_score"]) + 1)
plt.plot(n_features_range, rfecv.cv_results_["mean_test_score"], "o-")
plt.fill_between(
    n_features_range,
    rfecv.cv_results_["mean_test_score"] - rfecv.cv_results_["std_test_score"],
    rfecv.cv_results_["mean_test_score"] + rfecv.cv_results_["std_test_score"],
    alpha=0.2
)
plt.axvline(rfecv.n_features_, color="red", linestyle="--",
            label=f"Optimal: {rfecv.n_features_} features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validated ROC-AUC")
plt.title("RFECV: Optimal Number of Features")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 3. Embedded Methods — Feature Selection During Training

### Lasso (L1 Regularization)

Lasso shrinks coefficients to exactly zero, performing automatic feature selection.

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_regression(n_samples=500, n_features=30, n_informative=10,
                        noise=20, random_state=42)

# Scale before Lasso!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LassoCV automatically selects optimal alpha via CV
lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1)
lasso_cv.fit(X_scaled, y)

print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
print(f"Non-zero features: {(lasso_cv.coef_ != 0).sum()} / {X.shape[1]}")
zero_feats = np.where(lasso_cv.coef_ == 0)[0]
print(f"Features zeroed out: {len(zero_feats)}")

# Use as a selector
from sklearn.feature_selection import SelectFromModel
lasso_selector = SelectFromModel(lasso_cv, prefit=True)
X_lasso = lasso_selector.transform(X_scaled)
print(f"Shape after Lasso selection: {X_lasso.shape}")
```

---

### Tree-Based Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

X_cls, y_cls = make_classification(n_samples=1000, n_features=20,
                                    n_informative=8, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_cls, y_cls)

# MDI importance (mean decrease in impurity) — built-in, fast but biased
mdi_importance = pd.DataFrame({
    "feature":    [f"f_{i}" for i in range(20)],
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print("MDI Feature Importances (Top 10):")
print(mdi_importance.head(10))

# Permutation importance — model-agnostic, slower, less biased
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)
rf.fit(X_train, y_train)

perm_imp = permutation_importance(rf, X_test, y_test,
                                   n_repeats=20, random_state=42)
perm_df = pd.DataFrame({
    "feature":   [f"f_{i}" for i in range(20)],
    "importance": perm_imp.importances_mean,
    "std":        perm_imp.importances_std
}).sort_values("importance", ascending=False)
print("\nPermutation Importances (Top 10):")
print(perm_df.head(10))
```

---

### SHAP — Model-Agnostic Importance with Directionality

```python
# pip install shap
try:
    import shap
    from sklearn.ensemble import GradientBoostingClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)

    # Tree SHAP (fast for tree-based models)
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance (mean |SHAP|)
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":    [f"f_{i}" for i in range(20)],
        "mean_shap":  mean_shap
    }).sort_values("mean_shap", ascending=False)
    print("\nSHAP Feature Importances:")
    print(shap_df.head(10))

    # Summary plot
    shap.summary_plot(shap_values, X_test,
                      feature_names=[f"f_{i}" for i in range(20)])
except ImportError:
    print("Install shap: pip install shap")
```

**MDI vs Permutation vs SHAP:**

| Method | Speed | Bias | Directionality | Model-Agnostic |
|--------|-------|------|----------------|----------------|
| MDI (built-in) | Fast | High (favors high-cardinality) | No | No |
| Permutation | Moderate | Low | No | Yes |
| SHAP | Slow–Moderate | Low | YES | Mostly |

---

## 4. Variance Inflation Factor (VIF) — Remove Multicollinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

def compute_vif(df):
    """Compute VIF for all features. VIF > 5-10 indicates multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

def remove_high_vif(df, threshold=5.0):
    """Iteratively remove feature with highest VIF until all VIF < threshold."""
    features = list(df.columns)
    while True:
        vif = compute_vif(df[features])
        max_vif = vif["VIF"].max()
        if max_vif < threshold:
            break
        worst_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
        print(f"Removing '{worst_feature}' (VIF={max_vif:.2f})")
        features.remove(worst_feature)
    return features

# Example
np.random.seed(42)
n = 300
X1 = np.random.randn(n)
X2 = X1 + 0.1 * np.random.randn(n)   # X2 is almost = X1 → high VIF
X3 = np.random.randn(n)               # Independent

df_vif = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
print("Initial VIF:")
print(compute_vif(df_vif))

kept_features = remove_high_vif(df_vif)
print(f"\nFeatures after VIF filtering: {kept_features}")
```

---

## 5. Complete Feature Selection Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

X_cls, y_cls = make_classification(n_samples=1000, n_features=50,
                                    n_informative=10, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectFromModel(
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        threshold="mean"  # Keep features with importance > mean importance
    )),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_cls, y_cls, cv=cv, scoring="roc_auc")
print(f"Pipeline CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 6. Complete Comparison Table

| Method | Category | Computational Cost | Handles Nonlinear | Feature Interaction | Leakage Risk |
|--------|----------|--------------------|-------------------|---------------------|--------------|
| Pearson correlation | Filter | Very Low | No | No | Low |
| Spearman correlation | Filter | Very Low | Monotonic | No | Low |
| Mutual Information | Filter | Low | Yes | No | Low |
| Chi-Square | Filter | Very Low | No | No | Low |
| RFE | Wrapper | High | Depends on model | Partial | Medium (if outside CV) |
| RFECV | Wrapper | Very High | Depends on model | Partial | Low (CV inside) |
| Lasso | Embedded | Low | No (linear) | No | Low (inside model) |
| Tree importance (MDI) | Embedded | Low | Yes | Yes | Low |
| Permutation importance | Embedded | Moderate | Yes | Yes | Low |
| SHAP | Embedded | Moderate–High | Yes | Yes | Low |
| VIF | Filter | Low | No | No | N/A |

---

## 7. Interview Q&A

**Q1: What is the difference between filter, wrapper, and embedded methods?**
> **Filter:** Rank features by statistical properties (correlation, MI) independent of any model — fast but ignores feature interactions. **Wrapper:** Use a model to score subsets of features (RFE, forward/backward selection) — considers interactions but expensive. **Embedded:** Feature selection happens during model training (Lasso, tree importance) — efficient and considers model's own structure.

**Q2: Why is MDI (impurity-based) importance biased?**
> MDI computes the total decrease in node impurity when a feature is used for splitting. It's biased toward high-cardinality features (features with many unique values offer more split opportunities) and continuous features. Permutation importance is less biased — it measures the decrease in model performance when a feature is randomly shuffled, making it model-agnostic and scale-invariant.

**Q3: How can feature selection cause data leakage?**
> If you select features using the full dataset (including test data) before cross-validation, the feature selection process has "seen" the test data. For example, selecting top-k features by correlation with the target on the full dataset — the test set's labels influenced which features were selected. Fix: perform feature selection inside each CV fold, or use a Pipeline.

**Q4: What is VIF and when do you use it?**
> VIF (Variance Inflation Factor) quantifies how much the variance of a coefficient is inflated due to multicollinearity with other features. VIF=1 means no correlation; VIF>5-10 indicates problematic multicollinearity. Use VIF when training linear models where multicollinearity inflates coefficient variances and makes inference unreliable. Tree models don't need VIF analysis.

**Q5: What advantages does SHAP have over tree feature importance?**
> SHAP provides: (1) **Direction** — positive or negative impact on prediction, not just magnitude. (2) **Per-sample explanations** — SHAP values show impact for each individual prediction. (3) **Model-agnostic** — works for any model via KernelSHAP. (4) **Consistent** — satisfies Shapley axioms; MDI doesn't have this consistency property. SHAP is slower but far more informative for model interpretation.

**Q6: How many features should you select?**
> There's no universal answer. Practical approaches: (1) Start with RFECV to get a cross-validated optimal count. (2) Use the "elbow" in a performance vs n_features curve. (3) Let SelectFromModel(threshold="mean") pick automatically. (4) Consider business constraints: interpretability, inference latency, maintenance cost. More features → higher overfitting risk; fewer features → faster inference and simpler model.

---

## 8. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Selection leakage** | Select features on full data before CV | Use Pipeline or select inside each fold |
| **MDI for high-cardinality** | MDI biased toward many-valued features | Use permutation importance or SHAP |
| **Dropping all correlated features** | Some correlated features still useful | Remove one from each correlated pair |
| **Ignoring interactions** | Filter methods miss feature pairs | Use embedded methods or domain knowledge |
| **Selecting too few features** | Underfitting by over-pruning | Validate with learning curves |
| **VIF for tree models** | Unnecessary — trees handle multicollinearity | VIF only needed for linear models |
| **Chi-square with negatives** | Chi-square requires non-negative values | MinMaxScale first |

---

## 9. Quick Reference Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────┐
│                FEATURE SELECTION CHEAT SHEET                       │
├──────────────┬─────────────────────────────────────────────────────┤
│ FILTER       │ Fast, model-independent, ignores interactions        │
│              │ Pearson: linear correlation                          │
│              │ MI: nonlinear, categorical too                       │
│              │ Chi²: classification, needs non-negative             │
│              │ VIF: multicollinearity for linear models             │
├──────────────┼─────────────────────────────────────────────────────┤
│ WRAPPER      │ Slow, considers feature subsets                      │
│              │ RFE: fixed k, eliminates weakest iteratively         │
│              │ RFECV: auto-selects k via cross-validation           │
├──────────────┼─────────────────────────────────────────────────────┤
│ EMBEDDED     │ Efficient, integrated with model training            │
│              │ Lasso: zeros out irrelevant features (linear)        │
│              │ Tree MDI: fast but biased high-cardinality           │
│              │ Permutation: slower, less biased                     │
│              │ SHAP: direction + magnitude + per-sample             │
├──────────────┴─────────────────────────────────────────────────────┤
│ ⚠️  ALWAYS perform feature selection inside CV loop / Pipeline     │
│ ⚠️  MDI biased → prefer SHAP or permutation for final models       │
│ VIF > 10 → remove; VIF 5-10 → investigate                         │
└────────────────────────────────────────────────────────────────────┘
```
