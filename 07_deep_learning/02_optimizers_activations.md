# Optimizers & Activations — Complete Interview Guide

## 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | Optimizers update model weights to minimize loss; activations introduce non-linearity |
| **WHY** | The right optimizer controls convergence speed and quality; activations determine what functions a network can learn |
| **WHEN** | Adam is the default; SGD+momentum for fine-tuning; activations chosen per layer role |
| **HOW** | Compute gradient of loss, apply optimizer update rule, repeat per batch |
| **PITFALLS** | Learning rate too high/low, wrong optimizer for task, forgetting lr schedule |

---

## Optimizer Derivations

### SGD (Stochastic Gradient Descent)
```
θₜ₊₁ = θₜ - η · ∇L(θₜ)
```
- Simple, noisy, can escape local minima
- **Best for**: Large datasets, convex problems, final fine-tuning of pretrained models

### SGD + Momentum
```
vₜ = γ·vₜ₋₁ + η·∇L(θₜ)
θₜ₊₁ = θₜ - vₜ
```
- γ typically 0.9; accumulates gradient direction to accelerate convergence
- **Best for**: When loss landscape has high curvature or noisy gradients

### RMSProp
```
E[g²]ₜ = β·E[g²]ₜ₋₁ + (1-β)·gₜ²
θₜ₊₁ = θₜ - η/√(E[g²]ₜ + ε) · gₜ
```
- Adapts learning rate per parameter; divides by running mean of squared gradients
- β=0.9, ε=1e-8
- **Best for**: RNNs, non-stationary objectives

### Adam (Adaptive Moment Estimation)
```
mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ          # 1st moment (mean)
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²          # 2nd moment (variance)
m̂ₜ = mₜ/(1-β₁ᵗ)                    # bias correction
v̂ₜ = vₜ/(1-β₂ᵗ)                    # bias correction
θₜ₊₁ = θₜ - η · m̂ₜ/(√v̂ₜ + ε)
```
- β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3
- **Best for**: Default choice for deep learning — fast, robust

### AdamW (Adam + Weight Decay)
```
θₜ₊₁ = θₜ - η · [m̂ₜ/(√v̂ₜ + ε) + λ·θₜ]
```
- Decouples L2 regularization from gradient update — better generalization
- **Best for**: Transformers, BERT fine-tuning

---

## Optimizer Comparison Table

| Optimizer | Speed | Memory | Hyperparams | Best Use |
|-----------|-------|--------|-------------|----------|
| SGD | Slow | Low | lr, momentum | CV fine-tuning |
| Adam | Fast | Medium | lr, β₁, β₂ | General DL default |
| AdamW | Fast | Medium | lr, λ | Transformers |
| RMSProp | Medium | Medium | lr, β | RNNs |
| LBFGS | Very fast | High | lr | Small datasets |

---

## Batch Normalization

**Intuition**: Normalizes layer inputs to have zero mean and unit variance, then learns scale (γ) and shift (β).

```
μ_B = (1/m) Σ xᵢ
σ²_B = (1/m) Σ (xᵢ - μ_B)²
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
yᵢ = γ·x̂ᵢ + β
```

**Benefits**: Allows higher learning rates, reduces sensitivity to initialization, mild regularization effect.

```python
import torch.nn as nn

class BatchNormNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),   # BN before activation
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)
```

---

## Dropout

**Intuition**: Randomly zero out neurons during training with probability p → prevents co-adaptation → acts as ensemble of sub-networks.

```python
import torch.nn as nn

class DropoutNet(nn.Module):
    def __init__(self, input_dim, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Applied AFTER activation
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# CRITICAL: switch modes
model.train()   # dropout active
model.eval()    # dropout disabled (uses full network with scaled weights)
```

---

## Learning Rate Schedules

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 1. ReduceLROnPlateau — reduce lr when metric stops improving
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# 2. Cosine Annealing
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)

# 3. Step LR — multiply lr by gamma every step_size epochs
scheduler3 = optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# 4. OneCycleLR — warm up then anneal (best for fast convergence)
scheduler4 = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, steps_per_epoch=100, epochs=30
)

# Usage example with ReduceLROnPlateau
val_loss = 0.5
scheduler1.step(val_loss)
```

---

## Complete PyTorch Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data preparation
X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

model = MLP()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
best_state = None

for epoch in range(50):
    # --- Training ---
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
        val_preds = (torch.sigmoid(model(X_val_t)) > 0.5).numpy()
        val_acc = (val_preds == y_val).mean()

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Restore best model
model.load_state_dict(best_state)
print(f"Best Val Loss: {best_val_loss:.4f}")
```

---

## Interview Q&A

**Q1: Why does Adam work better than plain SGD in practice?**
Adam maintains per-parameter adaptive learning rates using first and second moment estimates. This handles sparse gradients, different feature scales, and saddle points better than a single global learning rate in SGD.

**Q2: What is the difference between L2 regularization and weight decay in Adam?**
In SGD, L2 regularization and weight decay are mathematically equivalent. In Adam, they are not — L2 adds `λθ` to the gradient before adaptive scaling (less effective), while AdamW's weight decay adds `λθ` directly to the parameter update (true regularization).

**Q3: When should you use SGD over Adam?**
SGD+momentum often generalizes better on vision tasks (CV) when training from scratch with careful lr tuning. Adam converges faster but may converge to sharper minima. For fine-tuning transformers, AdamW is preferred.

**Q4: How does Batch Normalization help training?**
BN normalizes pre-activations to reduce internal covariate shift, allowing higher learning rates and reducing gradient vanishing. It also acts as a regularizer (mild dropout-like effect from mini-batch noise).

**Q5: What dropout rate should you use?**
- Input layers: 0.1–0.2
- Hidden layers: 0.3–0.5
- Avoid on output layer
- Higher dropout for larger models; reduce if underfitting

**Q6: What is gradient clipping and when is it needed?**
Clips gradient norm to a maximum value to prevent exploding gradients — critical for RNNs and transformers. Rule: `clip_grad_norm_(params, max_norm=1.0)`.

---

## Common Pitfalls

1. **Calling `scheduler.step()` in the wrong place** — ReduceLROnPlateau needs the metric; others step per epoch not per batch
2. **Forgetting `optimizer.zero_grad()`** — gradients accumulate across batches
3. **Using `model.train()` during evaluation** — batch norm and dropout behave differently
4. **Too high initial lr with Adam** — try 1e-3, 3e-4, or 1e-4
5. **No gradient clipping with RNNs** — exploding gradients will ruin training
6. **Not saving best checkpoint** — final epoch model is often not the best model

---

## Quick Reference Cheat Sheet

```
Optimizer Defaults:
  Adam:    lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8
  AdamW:   lr=1e-4, weight_decay=1e-2  (transformers)
  SGD:     lr=1e-2, momentum=0.9, weight_decay=1e-4

Activation Guide:
  Hidden layers:    ReLU (default), GELU (transformers)
  Binary output:    Sigmoid (or BCEWithLogitsLoss + no activation)
  Multiclass:       Softmax (or CrossEntropyLoss + no activation)
  Regression:       None (linear)

Regularization:
  Dropout:          0.3–0.5 hidden, 0.1 input
  Weight decay:     1e-4 to 1e-2
  Batch Norm:       Before activation, after linear

LR Schedules:
  Fast convergence:   OneCycleLR
  Plateau-based:      ReduceLROnPlateau (patience=5–10)
  Cosine:             CosineAnnealingLR (T_max=num_epochs)

Training Checklist:
  ✓ StandardScaler on input
  ✓ optimizer.zero_grad()
  ✓ loss.backward()
  ✓ clip_grad_norm_() for RNNs
  ✓ optimizer.step()
  ✓ scheduler.step()
  ✓ model.eval() + torch.no_grad() for inference
```
