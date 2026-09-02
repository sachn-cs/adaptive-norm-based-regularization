# Reproduction recipe

This page walks through reproducing one of the paper's simulation
experiments using **regulo**.

## Install

```bash
git clone https://github.com/sachncs/regulo
cd regulo
pip install -e .
```

## Single-run quick check

```python
import numpy as np
from regulo import (
    Adam, ElasticNet, Mse, MLP, Runner, Square, synth,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, sigma_noise=0.10, seed=0)

# 75/25 split.
rng = np.random.default_rng(0)
perm = rng.permutation(200)
x_train, x_test = x[perm[:150]], x[perm[150:]]
y_train, y_test = y[perm[:150]], y[perm[150:]]

runner = Runner(
    MLP([20, 64, 32, 1], seed=0),
    Square(),
    ElasticNet(alpha=0.5, gamma=0.01),
    Adam(learning_rate=1e-3),
    batch_size=32,
    epochs=500,
)
runner.fit(x_train, y_train, seed=0)
print("test MSE:", Mse()(y_test, runner.predict(x_test)))
```

## Monte Carlo replications

The bundled demo script runs ``n_reps`` replications across all six
penalties and prints MSE mean / standard deviation:

```bash
python demo/run_simulation.py --seed 0           # 5 reps, ~1 minute
python demo/run_simulation.py --seed 0 --full    # 100 reps, slower
```

The demo uses a reduced hyperparameter grid for speed.  For the
full ``{0.001, 0.01, 0.1, 0.5, 0.9}`` grid from the paper, edit
``SIM_GRID`` in ``demo/run_simulation.py``.

## Save / load a trained model

```python
from regulo.store import save, load

save(runner, "model_dir")   # writes meta.json + weights/biases/adam .npz
restored = load("model_dir")
```

The on-disk format is plain ``npz`` + ``json`` (no ``pickle``), so
loading cannot execute arbitrary code.

## Cross-validation

```python
from regulo import (
    Adam, MLP, Ridge, Runner, Scalar, Square, search,
)

best_params, best_score = search(
    x_train, y_train,
    layer_sizes=[20, 64, 32, 1],
    method="ridge",
    param_grid=[{"lambda_": v} for v in (1e-3, 1e-2, 1e-1)],
    loss_fn=Square(),
    n_splits=5,
    epochs=200,
    seed=0,
)
```

``best_score`` is ``-MSE`` (higher is better) for regression and
balanced accuracy for classification.
