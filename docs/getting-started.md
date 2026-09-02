# Getting Started

This guide will walk you through installing and running the first experiment
with Adaptive Norm-Based Regularization.

## Prerequisites

- Python 3.10 or later
- pip (Python package installer)
- Git

## Quick Install

```bash
# Clone the repository
git clone https://github.com/sachncs/adaptive-norm-based-regularization.git
cd adaptive-norm-based-regularization

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# All 81 tests should pass
```

## Your First Experiment

### Run a Quick Simulation

The simplest way to see the project in action:

```bash
python demo/run_simulation.py
```

This runs a single replication of the paper's simulation experiments (Tables 1-3),
comparing all 6 regularizers on synthetic data.

### Run on Real Data

```bash
python demo/run_real_data.py
```

This applies the regularizers to the UCI Energy Efficiency dataset and
GSE9476 leukemia dataset.

### Programmatic Usage

```python
import numpy as np
from regulo.network import Network
from regulo.losses import MSELoss
from regulo.optimizer import Adam
from regulo.regularizers import Covridge
from regulo.trainer import Trainer

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 10)
true_weights = np.random.randn(10, 1)
y = X @ true_weights + 0.1 * np.random.randn(200, 1)

# Build a feedforward network
net = Network(
    layer_sizes=[10, 64, 32, 1],
    activations=["relu", "relu", "linear"],
)

# Configure training
loss_fn = MSELoss()
optimizer = Adam(net.parameters(), lr=1e-3)
regularizer = Covridge(lam1=0.01, lam2=0.001)

# Train
trainer = Trainer(net, loss_fn, optimizer, regularizer=regularizer)
history = trainer.fit(X, y, epochs=500, batch_size=32)

# Make predictions
predictions = net.forward(X[:5])
print("Predictions shape:", predictions.shape)
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation of all modules
- See [Architecture](architecture.md) for design decisions and data flow
- Check [Usage Guide](usage.md) for advanced training configurations
- Review [Fidelity Report](fidelity.md) for paper-to-code comparison

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'regulo'`, ensure you've
installed in editable mode:

```bash
pip install -e ".[dev]"
```

### Test Failures

If tests fail, check that your Python version is 3.10+:

```bash
python --version
```

### Performance Issues

For large-scale experiments, ensure NumPy is using optimized BLAS:

```bash
python -c "import numpy; numpy.show_config()"
```
