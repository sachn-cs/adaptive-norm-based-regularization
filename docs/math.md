# Mathematical Foundations

This document restates the core mathematics from the paper and maps each equation to its implementation.

---

## 1. Notation

- Bold lowercase: vectors `x ∈ R^p`
- Bold uppercase: matrices `W ∈ R^{p×q}`
- `‖x‖_q`: ℓ_q norm
- `‖A‖_F`: Frobenius norm
- `⟨A, B⟩ = tr(A^T B)`: trace inner product

---

## 2. Regularized Objective

The generic regularized objective for a neural network with parameters θ is:

```
J̃(θ; X, y) = J(θ) + λ Ω(W)
```

where `J(θ)` is the empirical loss (MSE or cross-entropy) and `Ω(W)` is a penalty on the weight matrices.

**Implementation:** `regulo/trainer.py` computes `total_loss = loss + reg_penalty` for each mini-batch.

---

## 3. Standard Penalties

### 3.1 Ridge

```
Ω_ridge(W) = ‖W‖_F^2 = Σ_{i,j} w_{ij}^2
```

Gradient: `∇_W Ω_ridge = 2 W`

**Implementation:** `regulo/regularizers.py::Ridge`

### 3.2 Lasso

```
Ω_lasso(W) = ‖W‖_1 = Σ_{i,j} |w_{ij}|
```

Subgradient: `∇_W Ω_lasso = sign(W)` (with `sign(0)=0`)

**Implementation:** `regulo/regularizers.py::Lasso`

### 3.3 Elastic Net

```
Ω_en(W) = α γ ‖W‖_1 + (1-α)/2 ‖W‖_F^2
```

**Implementation:** `regulo/regularizers.py::ElasticNet`

---

## 4. Geometry-Aware Penalties

### 4.1 Empirical Gram Matrix

For a representation matrix `H ∈ R^{n×p}`:

```
C_n = (1/n) H^T H
C_{δ,n} = C_n + δ I_p,   δ > 0
```

**Implementation:** `regulo/cv.py::build_regularizer` computes `c_n = (x_train.T @ x_train) / n` and `c_delta_n = c_n + delta * np.eye(p)`.

### 4.2 Covridge

```
Ω_covridge(W) = λ_1 ‖C_{δ,n}^{1/2} W‖_F^2 + λ_2 ‖W‖_F^2
```

Using the spectral decomposition `C_{δ,n} = U Λ U^T`, the penalty becomes:

```
Ω_covridge(W) = Σ_{i=1}^p (λ_1 (μ_{i} + δ) + λ_2) ‖w̃_i‖_2^2
```

where `w̃_i` is the i-th row of `Ũ = U^T W`.

Gradient:
```
∇_W Ω_covridge = 2 λ_1 C_{δ,n} W + 2 λ_2 W
```

**Implementation:** `regulo/regularizers.py::Covridge`
- `_c_sqrt` is precomputed via eigendecomposition.
- `penalty` computes `‖C^{1/2} W‖_F^2` directly.
- `gradient` computes `2 λ_1 C W + 2 λ_2 W` using `C = _c_sqrt @ _c_sqrt`.

### 4.3 Sparridge

```
Ω_sparridge(W) = λ_1 ‖C_{δ,n}^{1/2} W‖_F^2 + γ ‖W‖_1
```

Gradient (subgradient):
```
∇_W Ω_sparridge = 2 λ_1 C_{δ,n} W + γ sign(W)
```

**Implementation:** `regulo/regularizers.py::Sparridge`

---

## 5. Network Forward Pass

For a feedforward ReLU network with `L` hidden layers:

```
z^{(1)} = W^{(1)} x + b^{(1)}
h^{(1)} = ReLU(z^{(1)})
...
h^{(L)} = ReLU(z^{(L)})
ŷ = z^{(L+1)} = W^{(L+1)} h^{(L)} + b^{(L+1)}
```

**Implementation:** `regulo/network.py::FullyConnectedNetwork.forward`

---

## 6. Backpropagation

Given the gradient of the loss w.r.t. the output `δ^{(L+1)} = ∂J/∂ŷ`, the backpropagation rules are:

```
∂J/∂W^{(l)} = (1/n) (h^{(l-1)})^T δ^{(l)}
∂J/∂b^{(l)} = (1/n) Σ_i δ_i^{(l)}
δ^{(l-1)} = (δ^{(l)} (W^{(l)})^T) ⊙ I(z^{(l-1)} > 0)
```

**Implementation:** `regulo/network.py::FullyConnectedNetwork.backward`

**Important design choice:** The `1/n` factor is included in the loss backward pass (`MSELoss.backward` and `CrossEntropyLoss.backward`), so the network backward pass computes the unscaled matrix product `a_prev.T @ delta`.

---

## 7. Adam Optimizer

For each parameter `p` and gradient `g`:

```
m_t = β_1 m_{t-1} + (1-β_1) g_t
v_t = β_2 v_{t-1} + (1-β_2) g_t^2
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)
p_{t+1} = p_t - η m̂_t / (√v̂_t + ε)
```

**Implementation:** `regulo/optimizer.py::Adam`

---

## 8. Loss Functions

### 8.1 Mean Squared Error

```
J_MSE = (1/n) Σ_i (ŷ_i - y_i)^2
∂J/∂ŷ = (2/n) (ŷ - y)
```

**Implementation:** `regulo/losses.py::MSELoss`

### 8.2 Softmax Cross-Entropy

```
J_CE = -(1/n) Σ_i log( p_{i, y_i} )
where p_{i,c} = exp(z_{i,c}) / Σ_{c'} exp(z_{i,c'})
∂J/∂z = (1/n) (P - Y_onehot)
```

**Implementation:** `regulo/losses.py::CrossEntropyLoss`

---

## 9. Data Generating Processes

### 9.1 Covariance Structure

Informative predictors `x_{1:k}` are drawn from `N(0, Σ)` where:

```
Σ_{ii} = 1,   Σ_{ij} = ρ  (i ≠ j)
```

Noise predictors `x_{k+1:p}` are i.i.d. standard normal.

### 9.2 Response Models

Linear:
```
y = Σ_{j=1}^k θ_j^* x_j + ε,   ε ~ N(0, σ^2)
```

Nonlinear:
```
y = Σ_{j=1}^k θ_j^* sin(x_j) + ε
```

**Implementation:** `regulo/data.py::make_dgp`

---

## 10. Theoretical Results (Not Implemented)

### Theorem 5.1 (Covridge)

Under fixed-design assumptions A1–A3, the Covridge estimator is asymptotically Gaussian with sandwich covariance involving `Q`, `C_δ`, and the tuning parameters.

### Theorem 5.2 (Sparridge)

Under assumptions A1–A4, the scaled estimation error converges to the minimizer of a random convex criterion. When the ℓ1 scaling vanishes (`γ=0`), the limit is Gaussian; when `γ>0`, the limit is non-Gaussian.

These theorems are **not implemented** in this reproduction because they describe asymptotic statistical properties, not algorithmic steps.
