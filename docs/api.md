# API Reference

## Regularizers (`regulo.regularizers`)

### `class Regularizer(ABC)`

Abstract base class for all regularizers.

#### `penalty(weights: np.ndarray) -> float`
Compute scalar penalty for a weight matrix.

#### `gradient(weights: np.ndarray) -> np.ndarray`
Compute gradient of penalty w.r.t. weights, same shape as input.

---

### `class NoRegularizer(Regularizer)`

No-op baseline. Always returns `0.0` and zero gradient.

---

### `class Ridge(Regularizer)`

**Constructor:** `Ridge(lambda_: float)`

Penalty: `λ ‖W‖_F^2`

Gradient: `2 λ W`

---

### `class Lasso(Regularizer)`

**Constructor:** `Lasso(gamma: float)`

Penalty: `γ ‖W‖_1`

Gradient: `γ sign(W)` (subgradient, `sign(0)=0`)

---

### `class ElasticNet(Regularizer)`

**Constructor:** `ElasticNet(alpha: float, gamma: float)`

Penalty: `α γ ‖W‖_1 + (1-α)/2 ‖W‖_F^2`

Gradient: `α γ sign(W) + (1-α) W`

---

### `class Covridge(Regularizer)`

**Constructor:** `Covridge(lambda1: float, lambda2: float, c_delta_n: np.ndarray)`

Penalty: `λ_1 ‖C_{δ,n}^{1/2} W‖_F^2 + λ_2 ‖W‖_F^2`

Gradient: `2 λ_1 C_{δ,n} W + 2 λ_2 W`

`c_delta_n` is the stabilized Gram matrix `(p, p)`.

---

### `class Sparridge(Regularizer)`

**Constructor:** `Sparridge(lambda1: float, gamma: float, c_delta_n: np.ndarray)`

Penalty: `λ_1 ‖C_{δ,n}^{1/2} W‖_F^2 + γ ‖W‖_1`

Gradient: `2 λ_1 C_{δ,n} W + γ sign(W)`

---

## Losses (`regulo.losses`)

### `class MSELoss`

#### `forward(y_pred: np.ndarray, y_true: np.ndarray) -> float`
Mean squared error.

#### `backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray`
Gradient `2 (y_pred - y_true) / n_samples`.

---

### `class CrossEntropyLoss`

#### `forward(logits: np.ndarray, y_true: np.ndarray) -> float`
Softmax cross-entropy with integer labels.

#### `backward(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray`
Gradient `(softmax(logits) - onehot(y_true)) / n_samples`.

---

## Network (`regulo.network`)

### `class FullyConnectedNetwork`

**Constructor:** `FullyConnectedNetwork(layer_sizes: List[int])`

Creates a feedforward network with Xavier uniform initialization. ReLU is used for all hidden layers; the output layer is linear.

#### `forward(x: np.ndarray) -> np.ndarray`
Forward pass. Caches pre-activations (`_zs`) and activations (`_as`) for backprop.

#### `backward(dloss_dy: np.ndarray) -> Dict[str, List[np.ndarray]]`
Backward pass given the gradient of the loss w.r.t. the network output. Returns gradients for `weights` and `biases`.

#### `get_params() -> Dict[str, List[np.ndarray]]`
Return deep copy of all parameters.

#### `set_params(params: Dict[str, List[np.ndarray]]) -> None`
Set parameters from a dictionary.

---

## Optimizer (`regulo.optimizer`)

### `class Adam`

**Constructor:** `Adam(learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)`

#### `step(params, grads) -> Dict[str, List[np.ndarray]]`
Perform one Adam update. Returns new parameters.

#### `reset() -> None`
Clear all moment estimates and step counter.

---

## Data (`regulo.data`)

### `make_dgp(n, p, k, rho, sigma_noise, tau=1.0, nonlinear=False, random_state=None)`
Generate synthetic data with the paper's DGP specification.

Returns `(x, y)` with shapes `(n, p)` and `(n, 1)`.

### `make_dgp1(rho=0.25, sigma_noise=0.10, nonlinear=False, random_state=None)`
Wrapper for DGP1: `n=200, p=20, k=10`.

### `make_dgp2(rho=0.25, sigma_noise=0.10, nonlinear=False, random_state=None)`
Wrapper for DGP2: `n=1000, p=200, k=100`.

### `make_dgp3(rho=0.25, sigma_noise=0.10, nonlinear=False, random_state=None)`
Wrapper for DGP3: `n=500, p=2000, k=100`.

### `load_energy_data(test_size=0.25, random_state=None)`
Load UCI Energy Efficiency dataset (cooling load target).

Returns `(x_train, x_test, y_train, y_test, scaler)`.

### `load_leukemia_data(n_features=2000, test_size=0.2, random_state=None)`
Load GSE9476 leukemia microarray data with ANOVA feature selection.

Returns `(x_train, x_test, y_train, y_test, scaler)`.

---

## Metrics (`regulo.metrics`)

### `mean_squared_error(y_true, y_pred) -> float`
### `mean_absolute_error(y_true, y_pred) -> float`
### `root_mean_squared_error(y_true, y_pred) -> float`
### `r2_score(y_true, y_pred) -> float`
### `balanced_accuracy_score(y_true, y_pred) -> float`

---

## Trainer (`regulo.trainer`)

### `class Trainer`

**Constructor:**
```python
Trainer(
    network: FullyConnectedNetwork,
    loss_fn: MSELoss | CrossEntropyLoss,
    regularizer: Regularizer,
    optimizer: Adam,
    batch_size: int = 32,
    epochs: int = 500,
    early_stopping: bool = False,
    patience: int = 10,
    verbose: bool = False,
)
```

#### `fit(x_train, y_train, x_val=None, y_val=None) -> None`
Train the network. If `x_val` and `y_val` are provided, tracks validation loss and supports early stopping.

#### `predict(x: np.ndarray) -> np.ndarray`
Generate predictions on new data.

---

## Cross-Validation (`regulo.cv`)

### `build_regularizer(method, hp, x_train, delta=1e-4) -> Regularizer`
Instantiate a regularizer from a method name and hyperparameter dict.

Supported methods: `none`, `ridge`, `lasso`, `elastic_net`, `covridge`, `sparridge`.

### `grid_search_cv(x, y, layer_sizes, method, param_grid, loss_fn, n_splits=5, batch_size=32, epochs=500, learning_rate=1e-3, early_stopping=False, patience=10, task="regression", random_state=None) -> Tuple[Dict[str, float], float]`
Run k-fold cross-validation over a hyperparameter grid. Returns `(best_params, best_score)`.
