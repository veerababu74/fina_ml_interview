# Feature Engineering — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | The process of transforming raw data into informative features that improve model performance |
| **WHY** | Models can only learn from what's in the feature matrix; good features encode domain knowledge and improve signal-to-noise ratio |
| **WHEN** | During the data preparation phase; iteratively based on model performance and error analysis |
| **HOW** | Encoding, scaling, imputation, feature creation (interactions, ratios, datetime), binning |
| **PITFALLS** | Leakage (using future/test information), encoding without CV, scaling before splitting |

---

## 1. Encoding Categorical Variables

### One-Hot Encoding (OHE)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    "color": ["red", "blue", "green", "red", "blue"],
    "size":  ["S", "M", "L", "XL", "M"],
    "label": [1, 0, 1, 1, 0]
})

# sklearn OHE
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
color_encoded = ohe.fit_transform(df[["color"]])
color_cols = ohe.get_feature_names_out(["color"])
color_df = pd.DataFrame(color_encoded, columns=color_cols)
print(color_df)

# pandas get_dummies (simple for EDA)
color_dummies = pd.get_dummies(df["color"], prefix="color", drop_first=True)
print(color_dummies)
```

**When to use OHE:**
- Nominal (unordered) categories: color, city, product_type
- Low cardinality (< ~50 unique values)
- Linear models that can't handle ordinal assumptions
- **Beware:** High-cardinality OHE creates sparse, high-dimensional matrices → use target encoding instead

---

### Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_features = ["size"]
ordinal_categories = [["S", "M", "L", "XL"]]  # Explicit order!

oe = OrdinalEncoder(categories=ordinal_categories)
df["size_encoded"] = oe.fit_transform(df[["size"]])
print(df[["size", "size_encoded"]])
```

**When to use Ordinal Encoding:**
- Ordered categories: size (S<M<L), education (primary<secondary<university), rating (low<medium<high)
- Tree-based models that can find the right split points regardless of scale
- **Never use for unordered categories** — model will learn false relationships

---

### Target Encoding with k-Fold (Prevents Leakage)

Target encoding replaces a category with the mean target value for that category. **Must use cross-fold** to prevent leakage (the category's own label influences its encoding).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def target_encode_kfold(X_train, y_train, X_test, col, n_splits=5, smoothing=300):
    """
    k-Fold target encoding to prevent leakage.
    smoothing: reduces effect of rare categories (pulls toward global mean)
    """
    global_mean = y_train.mean()
    X_train_encoded = X_train[col].copy().astype(float)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_train):
        # Compute stats from train fold only
        train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        
        # Group stats per category
        stats = y_train_fold.groupby(train_fold[col]).agg(["mean", "count"])
        stats.columns = ["mean", "count"]
        
        # Smoothed encoding: blend category mean with global mean
        stats["encoded"] = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)
        
        # Apply to validation fold
        X_train_encoded.iloc[val_idx] = (
            X_train.iloc[val_idx][col].map(stats["encoded"]).fillna(global_mean)
        )
    
    # For test set, use full training statistics
    full_stats = y_train.groupby(X_train[col]).agg(["mean", "count"])
    full_stats.columns = ["mean", "count"]
    full_stats["encoded"] = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    
    X_test_encoded = X_test[col].map(full_stats["encoded"]).fillna(global_mean)
    
    return X_train_encoded, X_test_encoded


# Usage example
np.random.seed(42)
n = 500
train_df = pd.DataFrame({
    "city": np.random.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"], n),
    "label": np.random.randint(0, 2, n)
})
test_df = pd.DataFrame({
    "city": np.random.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"], 100)
})

train_encoded, test_encoded = target_encode_kfold(
    train_df, train_df["label"], test_df, "city"
)
print("Target encoded (train sample):")
print(pd.DataFrame({"city": train_df["city"], "encoded": train_encoded}).drop_duplicates().head(10))
```

**Encoding comparison summary:**

| Encoding | Use For | Pros | Cons |
|----------|---------|------|------|
| OHE | Nominal, low-cardinality | No ordinal assumption | High dimensionality |
| Ordinal | Ordered categories | Compact, preserves order | Wrong for nominal |
| Target | High-cardinality nominal | Compact, informative | Needs CV to prevent leakage |
| Hash | Very high cardinality | Fixed dim, fast | Collisions, no interpretability |
| Binary | Medium cardinality | Log(n) columns | Less interpretable |

---

## 2. Handling Missing Values

### Missing Value Strategies

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

np.random.seed(42)
df = pd.DataFrame({
    "age":    [25., 30., np.nan, 45., np.nan, 60.],
    "income": [50000., np.nan, 80000., np.nan, 120000., 95000.],
    "score":  [0.8, 0.6, np.nan, 0.9, 0.7, np.nan],
    "label":  [1, 0, 1, 1, 0, 1]
})

print("Missing values:\n", df.isnull().sum())

# Strategy 1: Mean/Median imputation
mean_imputer = SimpleImputer(strategy="mean")
median_imputer = SimpleImputer(strategy="median")

# Strategy 2: KNN imputation (uses neighboring samples)
knn_imputer = KNNImputer(n_neighbors=3)

# Strategy 3: MICE — Multiple Imputation by Chained Equations
mice_imputer = IterativeImputer(max_iter=10, random_state=42)

X = df.drop("label", axis=1)
for name, imputer in [
    ("Mean", mean_imputer),
    ("KNN",  knn_imputer),
    ("MICE", mice_imputer)
]:
    X_imputed = imputer.fit_transform(X)
    print(f"\n{name} imputation result:")
    print(pd.DataFrame(X_imputed, columns=X.columns).round(2))
```

### Missingness Indicator Feature

Missing values themselves can be **informative** (e.g., missing income might indicate unemployment).

```python
def add_missing_indicators(df, cols=None):
    """Add binary indicator columns for each column with missing values."""
    if cols is None:
        cols = df.columns[df.isnull().any()]
    
    indicators = {}
    for col in cols:
        if df[col].isnull().any():
            indicators[f"{col}_is_missing"] = df[col].isnull().astype(int)
    
    return pd.concat([df, pd.DataFrame(indicators)], axis=1)

df_with_indicators = add_missing_indicators(df)
print("\nWith missingness indicators:")
print(df_with_indicators)
```

**Missing value strategy guide:**
```
MCAR (Missing Completely At Random):
  → Simple mean/median imputation OK
  → Small fraction: drop rows

MAR (Missing At Random, dependent on observed data):
  → KNN or MICE imputation
  → Add missingness indicator

MNAR (Missing Not At Random, dependent on missing value):
  → Add missingness indicator
  → Domain-specific imputation
  → Example: patients with very high income may not report it
```

---

## 3. Scaling — Which Algorithm Needs Which Scaler

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
)
import numpy as np

# StandardScaler: z = (x - mean) / std → mean=0, std=1
# MinMaxScaler:   z = (x - min) / (max - min) → [0, 1]
# RobustScaler:   z = (x - median) / IQR → robust to outliers

np.random.seed(42)
data = np.array([10, 20, 30, 40, 50, 1000]).reshape(-1, 1)  # 1000 is outlier

for name, scaler in [
    ("Standard", StandardScaler()),
    ("MinMax",   MinMaxScaler()),
    ("Robust",   RobustScaler()),
]:
    scaled = scaler.fit_transform(data)
    print(f"{name}: {scaled.flatten().round(2)}")
```

**Scaling requirement by algorithm:**

| Algorithm | Needs Scaling | Recommended Scaler | Notes |
|-----------|---------------|--------------------|-------|
| Linear/Logistic Regression | YES | StandardScaler | Regularization assumes equal scale |
| SVM | YES | StandardScaler | Kernel methods sensitive to scale |
| k-NN | YES | StandardScaler / MinMax | Distance-based |
| PCA | YES | StandardScaler | Variance-based |
| Neural Networks | YES | StandardScaler / MinMax | Gradient stability |
| Ridge/Lasso | YES | StandardScaler | Penalizes coefficients |
| Decision Tree | NO | — | Split points are scale-invariant |
| Random Forest | NO | — | Tree-based |
| XGBoost/LightGBM | NO | — | Tree-based |
| k-Means | YES | StandardScaler | Distance-based |

---

## 4. Feature Creation

### Ratios and Interactions

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "revenue":   np.random.exponential(100000, 500),
    "cost":      np.random.exponential(70000,  500),
    "customers": np.random.randint(100, 10000, 500),
    "employees": np.random.randint(5, 500, 500),
    "age":       np.random.randint(20, 70, 500),
    "income":    np.random.exponential(60000, 500),
})

# Ratios — often more informative than raw values
df["profit_margin"]       = (df["revenue"] - df["cost"]) / df["revenue"]
df["revenue_per_customer"] = df["revenue"] / df["customers"]
df["revenue_per_employee"] = df["revenue"] / df["employees"]
df["debt_to_income"]       = df["cost"] / df["income"]

# Interaction features — product of two features
df["age_income_interaction"] = df["age"] * df["income"]

print(df[["revenue", "cost", "profit_margin", "revenue_per_customer"]].head())
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Add polynomial terms for nonlinear relationships
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X = df[["age", "income"]].values
X_poly = poly.fit_transform(X)
print(f"Original features: {X.shape[1]}")
print(f"Polynomial features (degree=2): {X_poly.shape[1]}")
print("Feature names:", poly.get_feature_names_out(["age", "income"]))
```

### Datetime Feature Extraction

```python
df_ts = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=500, freq="H"),
    "sales": np.random.exponential(1000, 500)
})

def extract_datetime_features(df, col):
    """Extract useful features from a datetime column."""
    dt = df[col].dt
    features = pd.DataFrame({
        f"{col}_year":         dt.year,
        f"{col}_month":        dt.month,
        f"{col}_day":          dt.day,
        f"{col}_hour":         dt.hour,
        f"{col}_dayofweek":    dt.dayofweek,       # 0=Monday, 6=Sunday
        f"{col}_is_weekend":   (dt.dayofweek >= 5).astype(int),
        f"{col}_quarter":      dt.quarter,
        f"{col}_dayofyear":    dt.dayofyear,
        # Cyclical encoding (preserves periodicity)
        f"{col}_month_sin":    np.sin(2 * np.pi * dt.month / 12),
        f"{col}_month_cos":    np.cos(2 * np.pi * dt.month / 12),
        f"{col}_hour_sin":     np.sin(2 * np.pi * dt.hour / 24),
        f"{col}_hour_cos":     np.cos(2 * np.pi * dt.hour / 24),
    })
    return features

ts_features = extract_datetime_features(df_ts, "timestamp")
print(ts_features.head())
```

**Why cyclical encoding?** Month 12 and Month 1 are close in time, but encoding them as integers (12 vs 1) makes them far apart. Sine/cosine encoding wraps around correctly.

---

## 5. Binning: pd.cut vs pd.qcut

```python
import pandas as pd
import numpy as np

np.random.seed(42)
ages = pd.Series(np.random.lognormal(3.5, 0.5, 500).clip(18, 100).astype(int))

# pd.cut — fixed-width bins (equal intervals)
age_cut = pd.cut(ages, bins=[0, 25, 35, 50, 65, 100],
                 labels=["18-25", "26-35", "36-50", "51-65", "65+"])
print("pd.cut (fixed intervals):")
print(age_cut.value_counts().sort_index())

# pd.qcut — quantile-based bins (equal frequency)
age_qcut = pd.qcut(ages, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
print("\npd.qcut (equal frequency):")
print(age_qcut.value_counts().sort_index())
```

| Method | Bins | Use When |
|--------|------|----------|
| `pd.cut` | Fixed width | Domain-meaningful intervals (age groups, price tiers) |
| `pd.qcut` | Equal frequency | Reduce skewness, ensure each bin has enough samples |

**Why bin continuous features?**
- Captures nonlinear relationships in linear models
- Handles outliers by absorbing extreme values into boundary bins
- Enables interaction with categorical features (age_group × product_category)

---

## 6. Interview Q&A

**Q1: What's the danger of target encoding and how do you prevent it?**
> Target encoding replaces a category with the mean target value for that category. If done naively (fit on full training set), the model effectively "sees" the labels of the validation folds during training, causing leakage and overly optimistic performance. Prevention: Use k-fold target encoding — compute the encoding for each sample using only the folds that don't contain that sample. In scikit-learn, use TargetEncoder (sklearn 1.3+) or wrap in a Pipeline.

**Q2: When should you scale features and when is it unnecessary?**
> Scale when the algorithm is sensitive to feature magnitude: linear/logistic regression (regularization assumes equal scale), SVMs (kernel computation), k-NN and k-Means (distance-based), neural networks (gradient stability), PCA (variance-based). Do NOT need to scale: tree-based models (Random Forest, XGBoost, LightGBM, decision trees) — they use split points that are scale-invariant.

**Q3: How do you handle high-cardinality categorical features?**
> OHE creates too many columns. Better approaches: (1) Target encoding with k-fold CV to prevent leakage. (2) Hash encoding with fixed dimension. (3) Frequency encoding (replace with count/frequency of that category). (4) Grouping rare categories into an "Other" bucket before OHE. (5) Embeddings (for neural networks, treat as embedding lookup).

**Q4: Explain cyclical encoding for time features.**
> Standard integer encoding of cyclical features (month 1–12) makes month 12 and month 1 appear maximally distant (distance=11), when they're actually adjacent in time. Cyclical encoding maps the feature to two dimensions using sine and cosine: `sin(2π × month/12)` and `cos(2π × month/12)`. This creates a circular representation where month 12 and month 1 are adjacent. Apply to: hours (24), months (12), days of week (7).

**Q5: What's the difference between pd.cut and pd.qcut?**
> `pd.cut` creates bins of equal width (e.g., 0-20, 20-40, 40-60). Good when the interval itself is meaningful (e.g., age groups). `pd.qcut` creates bins with an equal number of observations in each (quantile-based). Good when the data is skewed and you want each bin to be well-represented for training.

**Q6: How do you create features for a time series prediction problem?**
> Key feature types: (1) Lag features: the value at time t-1, t-7, t-30. (2) Rolling window statistics: mean, std, min, max over the last N periods. (3) Datetime features: hour, day of week, month, is_holiday, is_weekend. (4) Cumulative features: cumulative sum, cumulative mean. (5) Difference features: first difference (y_t − y_{t-1}), seasonal difference. Always compute these with respect to the training set to avoid leakage.

---

## 7. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Scaling before split** | Fit scaler on full data → leakage | Fit scaler inside CV / Pipeline |
| **Naive target encoding** | Encode with own target → inflated scores | Use k-fold target encoding |
| **Ordinal encoding nominals** | False ordering relationships | Use OHE for unordered categories |
| **Integer encoding nominals** | Same as ordinal pitfall | Use OHE or target encoding |
| **High-cardinality OHE** | Sparse, high-dimensional, slow | Use target or hash encoding |
| **Dropping missingness** | Losing informative signal | Add missing indicator before imputing |
| **Linear time features** | Month=12 far from Month=1 | Use cyclical encoding |
| **Interaction explosion** | Polynomial(degree=3) on 100 features | Be selective; use domain knowledge |

---

## 8. Quick Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING CHEAT SHEET                      │
├────────────────────┬─────────────────────────────────────────────────┤
│ ENCODING           │                                                  │
│ Nominal (<50 cats) │ One-Hot Encoding (drop_first=True)               │
│ Ordered categories │ OrdinalEncoder with explicit order               │
│ High cardinality   │ Target encoding (k-fold) or frequency encoding   │
├────────────────────┼─────────────────────────────────────────────────┤
│ MISSING VALUES     │                                                  │
│ MCAR, small pct    │ Drop rows or mean/median impute                  │
│ MAR                │ KNN or MICE imputation + indicator               │
│ MNAR               │ Missingness indicator + domain imputation        │
├────────────────────┼─────────────────────────────────────────────────┤
│ SCALING            │                                                  │
│ Linear/SVM/kNN/NN  │ StandardScaler (z-score)                        │
│ With outliers      │ RobustScaler                                     │
│ [0, 1] needed      │ MinMaxScaler                                     │
│ Tree-based         │ No scaling needed                                │
├────────────────────┼─────────────────────────────────────────────────┤
│ BINNING            │                                                  │
│ Equal width        │ pd.cut(bins=[...])                               │
│ Equal frequency    │ pd.qcut(q=N)                                     │
├────────────────────┼─────────────────────────────────────────────────┤
│ DATETIME           │ Extract: hour, dow, month, quarter + cyclical    │
│ CREATION           │ Ratios, interactions, polynomial, differences    │
└────────────────────┴─────────────────────────────────────────────────┘
GOLDEN RULE: All feature engineering must happen INSIDE the CV Pipeline!
```
