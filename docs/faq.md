# Frequently Asked Questions

## General

### What is this project?

This is a faithful end-to-end pure Python reproduction of the arXiv paper
"Adaptive Norm-Based Regularization for Neural Networks" by Muhammad Qasim
and Farrukh Javed (Lund University). It implements all components from scratch
using NumPy.

### Why use pure NumPy instead of PyTorch/TensorFlow?

The goal is faithful reproduction and educational clarity. Using NumPy allows
each component (network, optimizer, regularizer, loss) to be implemented from
scratch, making the mathematics transparent and auditable.

### Can I use this in production?

This project is primarily a research reproduction. While the code is well-tested,
it is optimized for clarity rather than performance. For production use, consider
implementing the regularizers in your preferred deep learning framework.

## Installation

### What Python version do I need?

Python 3.10 or later is required.

### Do I need a GPU?

No. This is a pure NumPy implementation that runs on CPU only.

### How do I install development dependencies?

```bash
pip install -e ".[dev]"
pip install black isort mypy
```

## Usage

### How do I compare all regularizers?

Use the demo scripts:

```bash
python demo/run_simulation.py      # Synthetic data
python demo/run_real_data.py       # Real datasets
```

### How do I add a custom regularizer?

Implement the `Regularizer` interface:

```python
from regulo.regularizers import Regularizer

class MyRegularizer(Regularizer):
    def __init__(self, my_param: float):
        self.my_param = my_param

    def penalty(self, weights: np.ndarray) -> float:
        """Compute the regularization penalty."""
        return self.my_param * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute the gradient of the penalty with respect to weights."""
        return 2 * self.my_param * weights
```

### How do I change the network architecture?

Modify the layer sizes and activations:

```python
from regulo.network import Network

# 3-layer network with different widths
net = Network(
    layer_sizes=[10, 128, 64, 32, 1],
    activations=["relu", "relu", "relu", "linear"],
)
```

### What are Covridge and Sparridge?

- **Covridge**: A covariance-weighted ridge penalty that adapts shrinkage to
  the empirical geometry of the inputs
- **Sparridge**: Similar to Covridge but with an L1 sparsity term instead of
  L2

Both outperform standard ridge, lasso, and elastic net in correlated or
high-dimensional settings.

## Testing

### How do I run all tests?

```bash
pytest tests/ -v
```

### How many tests are there?

The test suite contains 81 tests across 7 test files covering:
- Regularizer penalties and gradients
- Network forward/backward propagation
- Finite-difference gradient checks
- Adam optimizer convergence
- Loss function correctness
- Metric calculations
- End-to-end integration

### How do I run a specific test?

```bash
pytest tests/test_regularizers.py -v
pytest tests/test_network.py -k "test_backward_shape" -v
```

## Development

### How do I format my code?

```bash
black .
isort .
```

### How do I check types?

```bash
mypy regulo tests demo
```

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## Troubleshooting

### Tests fail with import errors

Ensure you've installed in editable mode:

```bash
pip install -e ".[dev]"
```

### Slow performance

NumPy performance depends on BLAS configuration. Check with:

```bash
python -c "import numpy; numpy.show_config()"
```

### GSE9476 dataset not available

The GSE9476 dataset may not be available on OpenML. The demo script will
print instructions for manual download if this occurs.
