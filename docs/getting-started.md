# Getting Started

This guide walks through installing **regulo** and running the
first experiment.

## Prerequisites

* Python 3.10 or later
* pip

## Quick install

```bash
git clone https://github.com/sachncs/regulo
cd regulo
pip install -e .
```

For development (tests, doctest):

```bash
pip install -e ".[dev]"
```

## Verify installation

```bash
pytest tests/ -v
```

All tests should pass and coverage should exceed 95%.

## Your first experiment

### Run the bundled simulation

```bash
python demo/run_simulation.py --seed 0
```

This runs 5 replications of all six penalties on a small
synthetic DGP and prints MSE mean / standard deviation.

### Programmatic usage

```python
import numpy as np
from regulo import (
    Adam, MLP, Ridge, Runner, Square, synth, Mse,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, sigma_noise=0.10, seed=0)
rng = np.random.default_rng(0)
perm = rng.permutation(200)
x_train, x_test = x[perm[:150]], x[perm[150:]]
y_train, y_test = y[perm[:150]], y[perm[150:]]

runner = Runner(
    MLP([20, 64, 32, 1], seed=0),
    Square(),
    Ridge(lambda_=0.01),
    Adam(learning_rate=1e-3),
    batch_size=32,
    epochs=500,
)
runner.fit(x_train, y_train, seed=0)
print("test MSE:", Mse()(y_test, runner.predict(x_test)))
```

## Reproducing a paper experiment

See [docs/repro.md](docs/repro.md) for the full reproduction
recipe.

## Next steps

* [docs/api.md](docs/api.md) -- full API reference
* [docs/architecture.md](docs/architecture.md) -- module map and
  design decisions
* [docs/math.md](docs/math.md) -- equation-by-equation mapping to
  the paper
* [docs/limits.md](docs/limits.md) -- explicit limitations

## Troubleshooting

### Import errors

If you see ``ModuleNotFoundError: No module named 'regulo'``,
make sure you've installed the package:

```bash
pip install -e .
```

### Test failures

Tests require Python 3.10 or later and NumPy 1.23+:

```bash
python --version
python -c "import numpy; print(numpy.__version__)"
```

### Performance

For large-scale experiments, ensure NumPy is linked against an
optimised BLAS:

```bash
python -c "import numpy; numpy.show_config()"
```
