# Neural Networks Basics — Complete Interview Guide

## 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | Neural networks are layered computational graphs of neurons that learn non-linear mappings from input to output via gradient descent |
| **WHY** | They can approximate any continuous function (Universal Approximation Theorem), enabling complex pattern recognition |
| **WHEN** | Use when data is high-dimensional (images, text, audio) and feature engineering is infeasible |
| **HOW** | Stack layers of linear transformations + non-linear activations; train via backpropagation |
| **PITFALLS** | Vanishing gradients, overfitting, poor initialization, wrong activation function |

---

## Single Neuron

A single neuron computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Wᵀx + b
output = activation(z)
```

- **W** = weight vector, **b** = bias, **x** = input features
- The activation introduces non-linearity — without it, stacked layers collapse to a single linear layer.

---

## Activation Functions

### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
- **Use when**: Hidden layers in deep networks (default choice)
- **Pros**: No vanishing gradient for positive values, computationally fast
- **Cons**: Dying ReLU problem (neurons stuck at 0)

### Sigmoid
```
f(x) = 1 / (1 + e^(-x))   range: (0, 1)
```
- **Use when**: Binary output layer
- **Cons**: Vanishing gradient, output not zero-centered

### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))   range: (-1, 1)
```
- **Use when**: Hidden layers in RNNs; better than sigmoid (zero-centered)
- **Cons**: Still suffers vanishing gradient

### Softmax
```
f(xᵢ) = e^xᵢ / Σⱼ e^xʲ
```
- **Use when**: Multi-class output layer (converts logits to probabilities)

### Leaky ReLU
```
f(x) = x if x > 0 else αx   (α typically 0.01)
```
- **Use when**: Fixing dying ReLU problem

---

## Forward Propagation

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(X, weights, biases):
    """
    X: input (batch_size, n_features)
    Returns activations at each layer
    """
    activations = [X]
    current = X
    for W, b in zip(weights[:-1], biases[:-1]):
        z = current @ W + b
        current = relu(z)
        activations.append(current)
    # Output layer with sigmoid
    z = current @ weights[-1] + biases[-1]
    current = sigmoid(z)
    activations.append(current)
    return activations

# Example
np.random.seed(42)
X = np.random.randn(100, 10)
W1 = np.random.randn(10, 64) * 0.01
b1 = np.zeros(64)
W2 = np.random.randn(64, 1) * 0.01
b2 = np.zeros(1)
acts = forward_pass(X, [W1, W2], [b1, b2])
print("Output shape:", acts[-1].shape)
```

---

## Backpropagation & Chain Rule

Loss gradient flows backward via chain rule:

```
∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
```

```python
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def backward_pass(X, y, weights, biases, lr=0.01):
    batch_size = X.shape[0]
    acts = forward_pass(X, weights, biases)
    
    # Output layer gradient (BCE + sigmoid combined)
    delta = acts[-1] - y.reshape(-1, 1)  # (batch, 1)
    
    grads_W = []
    grads_b = []
    
    for i in reversed(range(len(weights))):
        dW = acts[i].T @ delta / batch_size
        db = np.mean(delta, axis=0)
        grads_W.insert(0, dW)
        grads_b.insert(0, db)
        if i > 0:
            delta = delta @ weights[i].T
            delta[acts[i] <= 0] = 0  # ReLU backward
    
    # Update weights
    for i in range(len(weights)):
        weights[i] -= lr * grads_W[i]
        biases[i] -= lr * grads_b[i]
    return weights, biases
```

---

## Vanishing Gradient Problem

**Problem**: In deep networks, gradients shrink exponentially as they propagate back through sigmoid/tanh layers → early layers learn very slowly.

**Mathematical cause**: sigmoid derivative max is 0.25, so chaining 10 layers: 0.25¹⁰ ≈ 0.000001

**Fixes**:
1. **ReLU**: derivative is 1 for positive values — no shrinking
2. **Batch Normalization**: normalizes layer inputs
3. **Residual connections (ResNet)**: skip connections preserve gradient flow
4. **Gradient clipping**: cap gradient magnitude

```python
# Gradient clipping example (PyTorch)
import torch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## sklearn MLPClassifier

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (critical for neural nets)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42
)
mlp.fit(X_train, y_train)
print(classification_report(y_test, mlp.predict(X_test)))
```

---

## PyTorch TabularNet

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)

# Model
class TabularNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(1)

# Train
model = TabularNet(20, [128, 64])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(20):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(X_test_t)) > 0.5).numpy()
accuracy = (preds == y_test).mean()
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Interview Q&A

**Q1: Why do we need activation functions?**
Without non-linear activations, stacking multiple linear layers is equivalent to a single linear transformation — the network cannot learn complex patterns.

**Q2: What is the Universal Approximation Theorem?**
A feedforward network with at least one hidden layer and a non-linear activation can approximate any continuous function on a compact domain to arbitrary precision, given enough neurons.

**Q3: Why is weight initialization important?**
Bad initialization (e.g., all zeros) causes symmetry — all neurons compute the same gradient and never differentiate. Xavier/He initialization sets variance based on layer size.

**Q4: What is the dying ReLU problem?**
If a neuron's input is always negative, ReLU always outputs 0, gradient is 0, and the neuron never updates. Fix: Leaky ReLU, He initialization, lower learning rates.

**Q5: Explain the vanishing gradient problem.**
In deep networks with sigmoid/tanh, gradients are multiplied by small values (<1) at each layer. After many layers, gradients approach zero, preventing learning in early layers. ReLU and residual connections mitigate this.

**Q6: What's the difference between a validation set and a test set?**
The validation set is used during training to tune hyperparameters and monitor overfitting. The test set is held out completely and used only once for final performance estimation.

**Q7: How does batch size affect training?**
Small batches: noisy gradients (regularizing effect, may escape local minima). Large batches: stable gradients but may converge to sharp minima with worse generalization. Common: 32–256.

---

## Common Pitfalls

1. **Not scaling inputs**: Neural nets are sensitive to feature scale — always StandardScaler or MinMaxScaler
2. **Wrong output activation**: sigmoid for binary, softmax for multiclass, none for regression
3. **Learning rate too high**: loss diverges or oscillates
4. **No dropout or regularization**: overfitting on training data
5. **Forgetting `model.eval()`**: dropout/batch norm behave differently at test time
6. **Comparing raw logits vs probabilities**: use `BCEWithLogitsLoss` not `BCELoss` with logits

---

## Quick Reference Cheat Sheet

```
Architecture Tips:
  - Start with 2-3 hidden layers
  - Width: 64–512 neurons per layer
  - Hidden layers: ReLU (default)
  - Output: sigmoid (binary), softmax (multiclass), linear (regression)

Initialization:
  - He init for ReLU networks
  - Xavier for tanh/sigmoid

Regularization:
  - Dropout: 0.2–0.5
  - L2 weight decay: 1e-4 to 1e-2
  - Early stopping on validation loss

Optimizer defaults:
  - Adam lr=1e-3 (good default)
  - SGD+momentum lr=1e-2, momentum=0.9

Key formulas:
  Forward:  a = activation(Wx + b)
  BCE loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  Chain rule: ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```
