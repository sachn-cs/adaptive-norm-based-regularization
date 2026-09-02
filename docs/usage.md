# Usage Guide

## Setup

Install in a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

All tests:

```bash
pytest tests/ -v
```

Specific test files:

```bash
pytest tests/test_regularizers.py -v
pytest tests/test_network.py -v
pytest tests/test_integration.py -v
```

## Training a Model Manually

### Regression Example

```python
import numpy as np
from regulo.data import make_dgp1
from regulo.network import FullyConnectedNetwork
from regulo.optimizer import Adam
from regulo.regularizers import Ridge
from regulo.trainer import Trainer
import regulo.losses as losses

# Data
x, y = make_dgp1(rho=0.25, sigma_noise=0.1, random_state=42)
n = x.shape[0]
split = int(0.75 * n)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
net = FullyConnectedNetwork([20, 64, 32, 1])
opt = Adam(learning_rate=1e-3)
reg = Ridge(lambda_=0.01)
trainer = Trainer(
    net, losses.MSELoss(), reg, opt, batch_size=32, epochs=500
)
trainer.fit(x_train, y_train)

# Predict
preds = trainer.predict(x_test)
```

### Classification Example

```python
from regulo.losses import CrossEntropyLoss

net = FullyConnectedNetwork([p, 8, 4, n_classes])
opt = Adam(learning_rate=1e-3)
reg = Ridge(lambda_=1e-4)
trainer = Trainer(
    net, CrossEntropyLoss(), reg, opt,
    batch_size=16, epochs=500,
    early_stopping=True, patience=10
)
trainer.fit(x_train, y_train, x_val, y_val)
logits = trainer.predict(x_test)
class_preds = np.argmax(logits, axis=1)
```

## Hyperparameter Search

```python
from regulo.cv import grid_search_cv
import regulo.losses as losses

param_grid = [
    {"lambda1": a, "lambda2": b}
    for a in [0.001, 0.01, 0.1, 0.5, 0.9]
    for b in [0.001, 0.01, 0.1, 0.5, 0.9]
]

best_params, best_score = grid_search_cv(
    x_train, y_train,
    layer_sizes=[20, 64, 32, 1],
    method="covridge",
    param_grid=param_grid,
    loss_fn=losses.MSELoss(),
    n_splits=5,
    epochs=100,
)
```

## Simulation Experiments

```bash
cd demo
python run_simulation.py
```

For full 100 replications (slow):

```bash
python run_simulation.py --full
```

## Real-Data Experiments

```bash
cd demo
python run_real_data.py
```

## Custom Regularizer

Subclass `Regularizer` and implement `penalty` and `gradient`:

```python
from regulo.regularizers import Regularizer
import numpy as np

class HuberRegularizer(Regularizer):
    def __init__(self, delta: float):
        self.delta = delta

    def penalty(self, weights: np.ndarray) -> float:
        abs_w = np.abs(weights)
        quadratic = 0.5 * abs_w ** 2
        linear = self.delta * (abs_w - 0.5 * self.delta)
        return float(np.sum(np.where(abs_w <= self.delta, quadratic, linear)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        abs_w = np.abs(weights)
        return np.where(
            abs_w <= self.delta,
            weights,
            self.delta * np.sign(weights)
        )
```
