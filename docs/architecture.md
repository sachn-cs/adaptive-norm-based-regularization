# Architecture Documentation

## Overview

This package is a pure-Python, NumPy-based reproduction of the paper's empirical methodology. Every algorithmic component is implemented from scratch to ensure maximal fidelity and transparency.

---

## Module Map

```
regulo/
├── regularizers.py   # Penalty definitions and subgradients
├── losses.py         # MSE and cross-entropy with analytical derivatives
├── network.py        # Feedforward ReLU network, forward/backward
├── optimizer.py      # Adam with first and second moment estimates
├── data.py           # DGP generators and real-data loaders
├── metrics.py        # Evaluation metrics
├── trainer.py        # Epoch loop, mini-batching, early stopping
└── cv.py             # k-fold cross-validation and grid search
```

---

## Data Flow

```
[Data Source]
    │
    ▼
[make_dgp / load_energy_data / load_leukemia_data]
    │
    ▼
[Train / Val / Test Split]
    │
    ▼
[StandardScaler.fit_transform on train]
    │
    ▼
[C_{δ,n} Computation] ──→ [build_regularizer]
    │                         │
    │                         ▼
    │                   [Covridge / Sparridge / Ridge / Lasso / ElasticNet]
    │                         │
    ▼                         ▼
[FullyConnectedNetwork] ←── [Regularizer]
    │
    ▼
[Trainer.fit]
    │
    ├── Mini-batch sampling
    ├── Forward pass (caches z and a)
    ├── Loss computation
    ├── Regularizer penalty (first layer only for Covridge/Sparridge)
    ├── Backward pass (loss grad + regularizer grad)
    ├── Adam step
    └── Validation / early stopping
    │
    ▼
[Metrics: MSE, MAE, RMSE, R², balanced_accuracy]
```

---

## Key Design Decisions

### 1. Pure NumPy Backpropagation

Instead of using PyTorch or TensorFlow, we implement backpropagation manually. This ensures:
- Full control over regularizer gradient injection.
- Exact matching of the paper's update equations.
- No hidden framework defaults.

**Trade-off:** Slower than framework implementations, but fully transparent.

### 2. Layer-wise Regularization

The paper defines a single `C_{δ,n}` based on the input feature matrix. This matrix has shape `(p, p)`, matching the first-layer weight matrix `(p, h_1)`.

**Decision:** Covridge and Sparridge are applied **only to the first layer** (`W^{(1)}`). All other layers use the standard isotropic penalty (or no penalty, depending on the regularizer class).

This is documented as an assumption in `trainer.py` because the paper does not explicitly state how to handle multi-layer networks.

### 3. Loss Gradient Scaling

The `1/n_samples` factor is applied in `losses.py` (MSELoss.backward and CrossEntropyLoss.backward), not in `network.py`. This is consistent with standard deep learning frameworks and ensures the network backward pass computes the exact parameter gradients expected by the optimizer.

### 4. Adam State Management

The Adam optimizer maintains moment estimates in nested dictionaries keyed by parameter group name (`weights`, `biases`) and index. This allows it to handle arbitrary network depths without knowing the architecture ahead of time.

### 5. Cross-Validation Standardization

Feature standardization is performed **inside each CV fold** on the training portion only, then applied to the validation portion. This prevents data leakage.

### 6. Hyperparameter Grids

The simulation grid `{0.001, 0.01, 0.1, 0.5, 0.9}` matches the paper exactly. For two-parameter methods (Covridge, Sparridge, Elastic Net), all combinations are evaluated.

---

## Extensibility

The modular design allows easy extension:

- **New regularizer:** Subclass `Regularizer`, implement `penalty` and `gradient`, add to `build_regularizer`.
- **New loss:** Implement `forward` and `backward` methods.
- **New optimizer:** Implement `step(params, grads)`.
- **New data loader:** Add a function returning `(x_train, x_test, y_train, y_test, scaler)`.
