# 📈 Linear Regression — Complete Interview Guide

> **5-Pillar Answer Framework:** WHAT | WHY | WHEN | HOW | PITFALLS

---

## 🏛️ The 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | A parametric model that fits a hyperplane y = Xβ + ε to predict continuous outcomes by minimizing sum of squared residuals |
| **WHY** | Provides interpretable coefficients, closed-form solution exists, computationally efficient, strong theoretical backing |
| **WHEN** | Linear relationship between features and target, continuous target variable, need for interpretability, baseline modeling |
| **HOW** | OLS: β = (XᵀX)⁻¹Xᵀy or Gradient Descent; with regularization: Ridge (L2), Lasso (L1), ElasticNet |
| **PITFALLS** | Assumes linearity (check residuals), sensitive to outliers, multicollinearity inflates variance, requires feature scaling for GD |

---

## 📐 Mathematical Foundation

### Ordinary Least Squares (OLS)

The model: **y = Xβ + ε**

Where:
- `y` ∈ ℝⁿ — target vector (n samples)
- `X` ∈ ℝⁿˣᵖ — feature matrix (n samples, p features)
- `β` ∈ ℝᵖ — coefficient vector
- `ε` ∈ ℝⁿ — error/residual vector

**Loss Function (RSS — Residual Sum of Squares):**
```
L(β) = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)
     = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ
```

### Normal Equation Derivation

Take derivative with respect to β and set to zero:
```
∂L/∂β = -2Xᵀy + 2XᵀXβ = 0
XᵀXβ = Xᵀy
β = (XᵀX)⁻¹Xᵀy
```

**β = (XᵀX)⁻¹Xᵀy** ← The Normal Equation

> ⚠️ Requires XᵀX to be invertible (no perfect multicollinearity). Complexity: O(p³) for matrix inversion.

### Geometric Interpretation

OLS projects y onto the column space of X. The fitted values ŷ = Xβ̂ = X(XᵀX)⁻¹Xᵀy = Hy, where H = X(XᵀX)⁻¹Xᵀ is the **hat matrix** (projection matrix).

---

## 📋 Assumptions — LINE

| Letter | Assumption | How to Check | Fix if Violated |
|--------|-----------|--------------|----------------|
| **L** | **L**inearity | Residual vs Fitted plot | Transform features (log, sqrt, poly) |
| **I** | **I**ndependence | Durbin-Watson test | Use time-series model, GLS |
| **N** | **N**ormality of residuals | QQ-plot, Shapiro-Wilk | Log-transform target, robust regression |
| **E** | **E**qual variance (Homoscedasticity) | Scale-Location plot | Weighted LS, transform target |

**Additional:** No perfect multicollinearity (check VIF), No influential outliers (check Cook's distance)

---

## 💻 Implementation from Scratch — Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression


class LinearRegressionGD:
    """Linear Regression via Gradient Descent — from scratch."""

    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute MSE loss
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Compute gradients
            dw = -(2 / n_samples) * X.T @ (y - y_pred)
            db = -(2 / n_samples) * np.sum(y - y_pred)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Early stopping
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                print(f"Converged at iteration {i}")
                break

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        """R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# Demo
X, y = make_regression(n_samples=500, n_features=5, noise=15, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegressionGD(learning_rate=0.05, n_iterations=2000)
model.fit(X_scaled, y)
print(f"R² Score: {model.score(X_scaled, y):.4f}")
print(f"Weights: {model.weights}")
```

---

## 💻 sklearn: Ridge, Lasso, ElasticNet with Cross-Validation

```python
import numpy as np
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# Generate dataset
X, y = make_regression(
    n_samples=1000, n_features=20, n_informative=10,
    noise=20, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 1. Plain OLS ─────────────────────────────────────────────────
ols = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
ols_scores = cross_val_score(ols, X_train, y_train, cv=5, scoring='r2')
print(f"OLS R²: {ols_scores.mean():.4f} ± {ols_scores.std():.4f}")

# ── 2. Ridge with CV (auto alpha selection) ──────────────────────
alphas = np.logspace(-3, 3, 100)

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RidgeCV(alphas=alphas, cv=5, scoring='r2'))
])
ridge_pipeline.fit(X_train, y_train)
best_alpha_ridge = ridge_pipeline.named_steps['model'].alpha_
print(f"\nRidge best alpha: {best_alpha_ridge:.4f}")
print(f"Ridge Test R²: {ridge_pipeline.score(X_test, y_test):.4f}")

# ── 3. Lasso with CV ─────────────────────────────────────────────
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LassoCV(alphas=alphas, cv=5, max_iter=5000))
])
lasso_pipeline.fit(X_train, y_train)
best_alpha_lasso = lasso_pipeline.named_steps['model'].alpha_
lasso_coefs = lasso_pipeline.named_steps['model'].coef_
n_nonzero = np.sum(lasso_coefs != 0)
print(f"\nLasso best alpha: {best_alpha_lasso:.4f}")
print(f"Lasso non-zero features: {n_nonzero}/20")
print(f"Lasso Test R²: {lasso_pipeline.score(X_test, y_test):.4f}")

# ── 4. ElasticNet with CV ─────────────────────────────────────────
elasticnet_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=alphas, cv=5, max_iter=5000
    ))
])
elasticnet_pipeline.fit(X_train, y_train)
en_model = elasticnet_pipeline.named_steps['model']
print(f"\nElasticNet best alpha: {en_model.alpha_:.4f}")
print(f"ElasticNet best l1_ratio: {en_model.l1_ratio_:.2f}")
print(f"ElasticNet Test R²: {elasticnet_pipeline.score(X_test, y_test):.4f}")

# ── 5. Model Comparison ───────────────────────────────────────────
models = {
    'OLS': ols,
    'Ridge': ridge_pipeline,
    'Lasso': lasso_pipeline,
    'ElasticNet': elasticnet_pipeline
}

print("\n── Model Comparison ──")
print(f"{'Model':<15} {'Train R²':>10} {'Test R²':>10}")
print("-" * 37)
for name, model in models.items():
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"{name:<15} {train_r2:>10.4f} {test_r2:>10.4f}")
```

---

## 🔍 Diagnosing Assumption Violations

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pandas as pd


def diagnose_linear_regression(X, y, feature_names=None):
    """Comprehensive diagnostic plots and statistics for linear regression."""

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    standardized_residuals = residuals / np.std(residuals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Linear Regression Diagnostics', fontsize=14, fontweight='bold')

    # 1. Residuals vs Fitted (checks linearity & homoscedasticity)
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted\n(Should be random around 0)')

    # 2. QQ Plot (checks normality of residuals)
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot\n(Points should follow line)')

    # 3. Scale-Location Plot (checks homoscedasticity)
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location\n(Should be flat)')

    # 4. Residuals vs Leverage (Cook's Distance)
    n = len(y)
    p = X.shape[1] + 1
    hat_matrix_diag = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
    leverage = hat_matrix_diag
    cook_d = (standardized_residuals**2 / p) * (leverage / (1 - leverage))
    axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_title("Residuals vs Leverage\n(Watch for Cook's D > 1)")

    plt.tight_layout()
    plt.savefig('diagnostics.png', dpi=100, bbox_inches='tight')
    print("Diagnostic plots saved.")

    # VIF (Variance Inflation Factor) — multicollinearity check
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [
        variance_inflation_factor(X, i) for i in range(X.shape[1])
    ]
    vif_data["Status"] = vif_data["VIF"].apply(
        lambda v: "✅ OK" if v < 5 else ("⚠️ Moderate" if v < 10 else "❌ High")
    )

    print("\n── Variance Inflation Factors ──")
    print(vif_data.to_string(index=False))
    print("\nVIF > 10: Severe multicollinearity; VIF > 5: Moderate concern")

    # Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(residuals[:5000])  # limit for speed
    print(f"\n── Shapiro-Wilk Normality Test ──")
    print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    print("✅ Residuals appear normal" if p_value > 0.05 else "❌ Residuals not normal")


# Run diagnostics
X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
diagnose_linear_regression(X, y)
```

---

## ❓ Interview Questions & Answers

### Q1: When does OLS fail to find a unique solution?
**Answer:** When XᵀX is singular (non-invertible). This occurs when: (1) **Perfect multicollinearity** — one feature is a linear combination of others, (2) **More features than samples** (p > n) — the system is underdetermined. Solutions: Ridge regression (adds λI making it invertible), feature selection, PCA preprocessing.

### Q2: What's the difference between R² and Adjusted R²?
**Answer:** R² = 1 - SS_res/SS_tot, always increases as features are added (even if useless). Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1), penalizes for extra features. Always use Adjusted R² when comparing models with different numbers of features. R² is fine for a fixed feature set.

### Q3: Why use gradient descent instead of the normal equation?
**Answer:** Normal equation complexity is O(p³) for matrix inversion + O(np²) for XᵀX — infeasible for p > 10,000 features or n > millions. Gradient descent scales to large datasets (mini-batch SGD), works for non-linear extensions, and is the foundation for neural networks. Normal equation is preferred for small/medium datasets (p < 1,000).

### Q4: How do you detect and handle outliers in linear regression?
**Answer:** Detect via: Cook's Distance (> 4/n), DFFITS, studentized residuals (> 3). Handle by: (1) Investigate cause — data error or real? (2) Robust regression (Huber loss, RANSAC), (3) Log-transform skewed targets, (4) Remove if confirmed errors, (5) Use quantile regression if interested in specific percentiles.

### Q5: What is heteroscedasticity and why does it matter?
**Answer:** Heteroscedasticity = non-constant variance of residuals (e.g., variance increases with fitted values). It doesn't bias OLS coefficient estimates but makes standard errors incorrect, invalidating hypothesis tests and confidence intervals. Fix: log/sqrt transform target, weighted least squares, robust standard errors (HC3).

### Q6: Explain the Gauss-Markov theorem.
**Answer:** If LINE assumptions hold, OLS is the **B**est **L**inear **U**nbiased **E**stimator (BLUE) — it has the lowest variance among all linear unbiased estimators. "Best" means minimum variance. If we relax unbiasedness (add bias via regularization), we can get lower MSE — the bias-variance tradeoff.

### Q7: How do you interpret coefficients in multiple linear regression?
**Answer:** β_j = expected change in y for a one-unit increase in X_j, **holding all other features constant** (partial effect). This "ceteris paribus" interpretation requires no multicollinearity. For log-transformed features: β_j ≈ % change in y per 1% change in X_j. Standardized coefficients compare relative importance.

### Q8: What is the hat matrix and why is it important?
**Answer:** H = X(XᵀX)⁻¹Xᵀ projects y onto the column space of X: ŷ = Hy. Properties: symmetric, idempotent (H² = H), rank p. Diagonal elements h_ii (leverage) measure influence of observation i on its own prediction. High leverage points (h_ii > 2p/n) have unusual X values and can strongly influence estimates.

---

## ⚠️ Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Not scaling features | GD slow/diverges, coefficient magnitudes not comparable | StandardScaler before training |
| Multicollinearity | High VIF, unstable coefficients, wide CIs | Ridge, drop correlated features, PCA |
| Including target in features | Leakage, perfect R² | Careful feature engineering review |
| Extrapolating beyond training range | Inaccurate predictions | Warn users, collect more data |
| Ignoring residual plots | Violates assumptions, invalid inference | Always plot residuals |
| Forgetting intercept | Biased coefficients | `fit_intercept=True` (sklearn default) |
| Using R² alone | High R² with bad model | Always check RMSE, residual plots |

---

## 📋 Quick Reference Cheat Sheet

```
Linear Regression Quick Reference
══════════════════════════════════════════════════════════
Model:          y = Xβ + ε
Normal Eq:      β = (XᵀX)⁻¹Xᵀy
Loss (OLS):     L = Σ(yᵢ - ŷᵢ)²
GD Update:      β ← β - α · (2/n) · Xᵀ(Xβ - y)

Ridge:          L = RSS + λΣβⱼ²        → always unique solution
Lasso:          L = RSS + λΣ|βⱼ|       → sparse solution
ElasticNet:     L = RSS + λ[ρ|β|+(1-ρ)β²]  → combines both

Assumptions:    LINE (Linearity, Independence, Normality, Equal Var)
Metrics:        R², Adjusted R², RMSE, MAE
Diagnose:       Residual plot, QQ-plot, VIF, Cook's distance

When to use:    Continuous target, linear relationship, interpretability needed
When to avoid:  Non-linear patterns, too many irrelevant features (→ Lasso)

sklearn:
    from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
══════════════════════════════════════════════════════════
```
