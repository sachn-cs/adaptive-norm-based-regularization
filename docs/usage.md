# Usage Guide

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=regulo --cov-fail-under=95 --doctest-modules
```

Per-module:

```bash
pytest tests/test_penalty.py -v
pytest tests/test_net.py -v
pytest tests/test_fit.py -v
```

## Training a model manually

### Regression

```python
import numpy as np
from regulo import (
    Adam, MLP, Mse, Ridge, Runner, Scalar, Square, synth,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, sigma_noise=0.10, seed=42)
rng = np.random.default_rng(42)
perm = rng.permutation(200)
x_train, x_test = x[perm[:150]], x[perm[150:]]
y_train, y_test = y[perm[:150]], y[perm[150:]]

scaler = Scaler().fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

runner = Runner(
    MLP([20, 64, 32, 1], seed=42),
    Square(),
    Ridge(lambda_=0.01),
    Adam(learning_rate=1e-3),
    batch_size=32,
    epochs=500,
)
runner.fit(x_train_s, y_train, x_val=x_test_s, y_val=y_test, seed=42)
print(Mse()(y_test, runner.predict(x_test_s)))
```

### Classification

```python
import numpy as np
from regulo import (
    Adam, Balanced, Lasso, MLP, Runner, Scaler, Softmax,
)

rng = np.random.default_rng(0)
x = rng.standard_normal((200, 10))
y = rng.integers(0, 3, size=(200,))

scaler = Scaler().fit(x)
x_s = scaler.transform(x)

runner = Runner(
    MLP([10, 64, 32, 3], seed=0),
    Softmax(),
    Lasso(gamma=0.01),
    Adam(learning_rate=1e-3),
    batch_size=32,
    epochs=200,
)
runner.fit(x_s, y, seed=0)
preds = runner.predict_class(x_s)
print("balanced accuracy:", Balanced()(y, preds))
```

## Hyperparameter search

```python
from regulo import search

best, score = search(
    x, y,
    layer_sizes=[10, 64, 32, 3],
    method="lasso",
    param_grid=[{"gamma": v} for v in (1e-3, 1e-2, 1e-1)],
    loss_fn=Softmax(),
    n_splits=5,
    epochs=200,
    seed=0,
)
```

## Save and load a trained model

```python
from regulo.store import save, load

save(runner, "model_dir")
restored = load("model_dir")
```

The on-disk format is plain ``npz`` + ``json`` (no ``pickle``);
loading cannot execute arbitrary code.  Major-version mismatches
are rejected.
