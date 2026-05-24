# Regression Metrics — Complete Interview Guide

---

## 5-Pillar Answer Framework

| Pillar | Key Point |
|--------|-----------|
| **WHAT** | Quantitative measures for how closely continuous predictions match actual values |
| **WHY** | Different metrics penalize errors differently; business context determines the right choice |
| **WHEN** | MSE for gradient-based optimization; MAE for robustness; MAPE for business reporting; R² for explained variance |
| **HOW** | Compute residuals (y_true − y_pred), aggregate with different norms (L1, L2, relative) |
| **PITFALLS** | R² can be negative; MAPE breaks with zeros; MSE inflates large errors; Adjusted R² for feature comparison |

---

## 1. Core Regression Metrics

### MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Interpretation:** Average absolute deviation from the truth.
- **Unit:** Same as the target variable.
- **Robust to outliers:** Uses L1 norm — outliers affect it linearly, not quadratically.
- **Downside:** Not differentiable at zero → subgradient required; less convenient for gradient descent.

---

### MSE — Mean Squared Error

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Interpretation:** Average squared deviation.
- **Unit:** Squared units of target (less interpretable).
- **Sensitive to outliers:** Squaring amplifies large errors — a 10× error contributes 100× the penalty.
- **Advantage:** Smooth, differentiable → ideal for gradient-based optimization.

---

### RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Interpretation:** Same unit as the target → more interpretable than MSE.
- **Still sensitive to outliers** (just square-rooted MSE).
- **RMSE vs MAE:** RMSE ≥ MAE always. If RMSE >> MAE, large errors dominate — investigate outliers.

---

### MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- **Interpretation:** Average % deviation → scale-independent, easy to communicate to business.
- **Major pitfall:** Undefined when `y_i = 0`; asymmetric (large penalty for underestimation).
- **Alternative — sMAPE (Symmetric MAPE):**

$$\text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$

---

### R² — Coefficient of Determination

$$R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- **Interpretation:** Proportion of variance explained by the model.
- **Range:** (−∞, 1]. R²=1: perfect; R²=0: model = constant mean; **R² < 0: model is worse than predicting the mean**.
- **R² always increases (or stays the same) as you add features**, even useless ones → use Adjusted R².

---

### Adjusted R² — Penalizes Extra Features

$$\bar{R}^2 = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}$$

where:
- `n` = number of samples
- `p` = number of predictors (features)

- **Adjusted R² can decrease** when adding a feature that doesn't improve fit proportionally.
- **Use when comparing models with different numbers of features.**

---

### RMSLE — Root Mean Squared Log Error

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2}$$

- **Use when:** Target spans multiple orders of magnitude (housing prices, revenue, sales counts).
- **Effect:** Penalizes underestimation more than overestimation for positive targets.
- **Requires:** Both `y` and `ŷ` must be non-negative.

---

## 2. When to Use MSE vs MAE

| Situation | Prefer |
|-----------|--------|
| Outliers are informative and should drive optimization | **MSE** |
| Outliers are noise or measurement errors | **MAE** |
| Gradient-based optimization required | **MSE** (smooth everywhere) |
| Robust training without outlier influence | **MAE** |
| Business communication / reporting | **MAPE** or **MAE** |
| Features span multiple scales | **RMSLE** |
| Comparing models with different # features | **Adjusted R²** |

**Rule of thumb:** Start with RMSE for training. Validate with MAE for interpretability. Use MAPE if the business team needs %-based reporting.

---

## 3. R² Pitfalls in Detail

```python
import numpy as np
from sklearn.metrics import r2_score

np.random.seed(42)
y_true = np.array([3, -0.5, 2, 7, 4.2, 5.1])

# Perfect model
y_perfect = y_true.copy()
print(f"R² (perfect):   {r2_score(y_true, y_perfect):.4f}")  # 1.0

# Constant mean predictor
y_mean = np.full_like(y_true, y_true.mean())
print(f"R² (mean pred): {r2_score(y_true, y_mean):.4f}")  # 0.0

# Terrible model (worse than mean)
y_terrible = y_true + 50
print(f"R² (terrible):  {r2_score(y_true, y_terrible):.4f}")  # Very negative!

# Adding a useless feature increases R²!
from sklearn.linear_model import LinearRegression
X = np.random.randn(50, 1)
y = 2 * X[:, 0] + np.random.randn(50)

r2_list, adj_r2_list = [], []
for n_features in range(1, 15):
    X_extended = np.hstack([X, np.random.randn(50, n_features - 1)])  # add noise features
    model = LinearRegression().fit(X_extended, y)
    y_pred = model.predict(X_extended)
    r2 = r2_score(y, y_pred)
    n, p = 50, n_features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    r2_list.append(r2)
    adj_r2_list.append(adj_r2)
    print(f"Features: {n_features:2d} → R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")
```

**Output pattern:** R² monotonically increases (never decreases) as we add noise features. Adjusted R² correctly penalizes and may decrease — use it for model selection.

---

## 4. Complete Python Code — All Metrics

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
    mean_squared_log_error
)

# Generate regression dataset
X, y = make_regression(
    n_samples=500, n_features=10, noise=20, random_state=42
)
y = np.abs(y) + 100  # Make positive for RMSLE/MAPE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8} {'Adj R²':>8}")
print("-" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pos = np.maximum(y_pred, 0)  # Clip for RMSLE

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2   = r2_score(y_test, y_pred)
    n, p = len(y_test), X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"{name:<20} {mae:>8.2f} {rmse:>8.2f} {mape:>8.2f} {r2:>8.4f} {adj_r2:>8.4f}")


def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error"""
    y_pred_clipped = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred_clipped))


def adjusted_r2(r2, n, p):
    """Adjusted R² formula"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

---

## 5. Business Metric Conversion Examples

### Revenue Prediction
```python
# Translate model metrics to business impact
y_true_revenue = np.array([10000, 25000, 8000, 15000, 30000])
y_pred_revenue = np.array([9500,  27000, 7800, 14200, 32000])

mae = mean_absolute_error(y_true_revenue, y_pred_revenue)
mape = mean_absolute_percentage_error(y_true_revenue, y_pred_revenue) * 100

print(f"Average prediction error: ${mae:,.0f}")
print(f"Average % error:          {mape:.1f}%")
# Business translation: "Our model's predictions are off by ~$X on average,
# which represents Y% of typical deal size"
```

### Inventory Forecasting
```python
# Overstock vs understock costs
overstock_cost_per_unit = 2    # holding cost
understock_cost_per_unit = 10  # lost sale + expediting

errors = y_pred_revenue - y_true_revenue
overstock  = np.maximum(errors, 0)
understock = np.maximum(-errors, 0)

total_cost = (overstock * overstock_cost_per_unit +
              understock * understock_cost_per_unit).sum()
print(f"Total inventory cost from errors: ${total_cost:,.0f}")
```

---

## 6. Residual Analysis

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Residuals vs Predicted
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Predicted\n(look for patterns/funnels)")

# 2. Histogram of residuals
axes[1].hist(residuals, bins=30, edgecolor="k")
axes[1].set_xlabel("Residual")
axes[1].set_title("Residual Distribution\n(should be ≈ normal)")

# 3. Q-Q plot
import scipy.stats as stats
stats.probplot(residuals, plot=axes[2])
axes[2].set_title("Q-Q Plot\n(points on line = normal residuals)")

plt.tight_layout()
plt.show()
```

**What to look for:**
- **Funnel shape** in residuals vs predicted → Heteroscedasticity → try log-transforming target
- **Curved pattern** → Missing nonlinear terms → add polynomial features or use tree model
- **Heavy tails in Q-Q plot** → Outliers → investigate, consider robust regression

---

## 7. Interview Q&A

**Q1: When would you choose MAE over MSE as your optimization metric?**
> When the dataset contains outliers that you consider noise (not signal). MSE squares errors, so a single outlier can dominate gradients and pull the model toward fitting that point. MAE treats all errors proportionally, making it more robust. Example: housing price with a few typo entries of $1B should use MAE.

**Q2: Can R² be negative? What does that mean?**
> Yes. R² = 1 − SS_res/SS_tot. If the model performs worse than simply predicting the mean (SS_res > SS_tot), R² becomes negative. This indicates the model is actively harmful — it's better to predict the mean for every observation.

**Q3: Why does R² always increase with more features? How do you fix this?**
> R² measures fraction of variance explained. Adding any feature, even noise, can only increase (or maintain) the explained variance in-sample because the model has more degrees of freedom to fit the data. Fix: use Adjusted R², which penalizes extra parameters by multiplying (1-R²) by (n-1)/(n-p-1). Adjusted R² can decrease when useless features are added.

**Q4: When would you use RMSLE over RMSE?**
> RMSLE is appropriate when (1) the target spans orders of magnitude (e.g., from $100 to $1M), (2) relative errors matter more than absolute errors, and (3) underestimation is more costly than overestimation. Log-space errors treat a prediction of 1000 for a true 1100 equally to a prediction of 100 for a true 110.

**Q5: How do you translate model metrics to business impact?**
> Map the model metric to the decision it supports: (1) For inventory, convert MAE to overstock/understock cost using unit economics. (2) For revenue forecasting, express MAPE as "our forecasts are off by X% of deal value, which costs us $Y in misallocated resources." (3) Build a simulation: apply the model's error distribution to historical business outcomes.

**Q6: What's the difference between in-sample R² and out-of-sample R²?**
> In-sample R² (on training data) can be high even for overfit models. Out-of-sample (test set) R² measures true generalization. Always report out-of-sample R². A model with in-sample R²=0.99 and test R²=0.20 is severely overfit.

---

## 8. Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **MAPE with zeros** | Division by zero crashes or produces inf | Use sMAPE or WMAPE instead |
| **R² for comparison** | R² increases monotonically with features | Use Adjusted R² for model selection |
| **Negative R²** | Model worse than predicting mean | Debug model — check for target leakage, wrong objective |
| **RMSE units** | Reporting RMSE without context | Express as % of mean target or compare to baseline RMSE |
| **MSE with outliers** | Outliers dominate MSE-based loss | Inspect data; use MAE, Huber loss, or remove outliers |
| **In-sample evaluation** | Reporting training metrics only | Always evaluate on held-out test set |
| **Scale mismatch** | Comparing RMSE across different targets | Use R², MAPE, or normalized metrics for cross-target comparison |

---

## 9. Quick Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│                  REGRESSION METRICS CHEAT SHEET                      │
├──────────────┬──────────────────────────────┬────────────────────────┤
│ Metric       │ Formula                      │ Key Property           │
├──────────────┼──────────────────────────────┼────────────────────────┤
│ MAE          │ mean(|y - ŷ|)                │ Robust, interpretable  │
│ MSE          │ mean((y - ŷ)²)               │ Differentiable, outlier│
│ RMSE         │ √MSE                         │ Same unit as y         │
│ MAPE         │ mean(|y-ŷ|/|y|) × 100       │ %-based, no y=0        │
│ R²           │ 1 - SS_res/SS_tot            │ (-∞, 1], increases w/p │
│ Adjusted R²  │ 1-(1-R²)(n-1)/(n-p-1)       │ Penalizes extra feats  │
│ RMSLE        │ √mean((log(1+y)-log(1+ŷ))²) │ Good for skewed target │
├──────────────┴──────────────────────────────┴────────────────────────┤
│ RMSE vs MAE: Both in same unit; RMSE ≥ MAE always                    │
│ If RMSE >> MAE: large errors present — investigate outliers          │
│ R² = 0 → predicting mean; R² < 0 → worse than mean                  │
│ MAPE breaks with zero/near-zero targets → use sMAPE                  │
└──────────────────────────────────────────────────────────────────────┘
```

**Decision Flowchart:**
```
What matters more — absolute or relative errors?
  Absolute → MAE / RMSE
  Relative → MAPE / RMSLE

Are outliers noise?
  YES → MAE (robust)
  NO  → RMSE (penalizes outliers more)

Comparing models with different # features?
  YES → Adjusted R²
  NO  → R² or RMSE

Target spans multiple orders of magnitude?
  YES → RMSLE
  NO  → RMSE or MAE
```
