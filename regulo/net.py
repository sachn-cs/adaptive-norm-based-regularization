"""Manual feedforward ReLU network with back-propagation.

Implements a fully connected multi-layer perceptron using only
NumPy.  All forward and backward passes are explicit -- there is no
autograd engine.  Weight initialization follows the Xavier (Glorot)
uniform scheme.

Architecture
------------
The network is defined by a list of layer widths
``[input, hidden_1, ..., output]``.  Hidden layers use ReLU
activations; the output layer is linear (no activation).  This is
the standard architecture for regression; for classification the
output is passed through a softmax in the loss function (see
:class:`regulo.loss.Softmax`).

Back-propagation
----------------
The backward pass computes exact gradients w.r.t. all weights and
biases using the chain rule.  It requires cached activations from
the forward pass (stored in :attr:`MLP.cache_z` and
:attr:`MLP.cache_a`), so :meth:`MLP.__call__` must be invoked before
:meth:`MLP.grad`.
"""

from typing import Dict, List, Optional

import numpy as np

__all__ = ["MLP", "xavier"]


def xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a weight matrix from the Xavier uniform distribution.

    Draws each entry independently from ``Uniform(-limit, limit)``
    with ``limit = sqrt(6 / (fan_in + fan_out))``.  The caller
    supplies a NumPy :class:`~numpy.random.Generator` to make
    initialization reproducible across runs.

    Args:
        fan_in: Number of input features (``W.shape[0]``).
        fan_out: Number of output features (``W.shape[1]``).
        rng: NumPy random generator used for sampling.

    Returns:
        Weight matrix of shape ``(fan_in, fan_out)``.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class MLP:
    """Feedforward MLP with ReLU hidden activations and a linear output.

    The network stores its own forward-pass activations and
    pre-activation values in :attr:`cache_z` and :attr:`cache_a`
    respectively so that :meth:`grad` can compute exact gradients
    without re-running the forward pass.

    Attributes:
        layer_sizes: List of layer widths ``[input, hidden_1, ..., output]``.
        n_layers: Number of weight matrices (``len(layer_sizes) - 1``).
        weights: List of weight matrices, one per layer, in
            input-to-output order.
        biases: List of row-vector biases, one per layer.
        cache_z: Pre-activation values from the most recent forward
            pass.  Populated by :meth:`__call__`.
        cache_a: Activation values from the most recent forward
            pass.  Populated by :meth:`__call__`.
    """

    def __init__(self, layer_sizes: List[int], seed: Optional[int] = None) -> None:
        """Initialise weights with Xavier-uniform and biases to zero.

        Args:
            layer_sizes: Layer widths including input and output.
            seed: Optional integer seed for the NumPy random
                generator used by :func:`xavier`.  ``None`` defers to
                NumPy's default seeding.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements.")
        if any(s <= 0 for s in layer_sizes):
            raise ValueError("All layer widths must be positive.")
        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            self.weights.append(xavier(layer_sizes[i], layer_sizes[i + 1], rng))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        self.cache_z: List[np.ndarray] = []
        self.cache_a: List[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass, caching intermediates for :meth:`grad`."""
        self.cache_z = []
        self.cache_a = [x]
        a = x
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.cache_z.append(z)
            if i < self.n_layers - 1:
                a = np.maximum(z, 0.0)
            else:
                a = z
            self.cache_a.append(a)
        return a

    def grad(self, dloss_dy: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Compute parameter gradients via back-propagation.

        Assumes :meth:`__call__` has been invoked immediately before
        and that ``dloss_dy`` already contains the ``1 / n_samples``
        scaling from the loss backward pass.  The returned
        gradients are **not** scaled by the learning rate -- that
        is the optimizer's job.

        Returns:
            Dictionary with ``"weights"`` and ``"biases"`` lists,
            each in input-to-output order.
        """
        if not self.cache_a:
            raise RuntimeError("__call__ must be invoked before grad().")
        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []
        delta = dloss_dy
        for i in reversed(range(self.n_layers)):
            a_prev = self.cache_a[i]
            dw = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T
                z = self.cache_z[i - 1]
                delta *= (z > 0.0).astype(delta.dtype)
        return {"weights": grads_w, "biases": grads_b}

    def state(self) -> Dict[str, List[np.ndarray]]:
        """Return deep copies of all current parameters."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def load(self, params: Dict[str, List[np.ndarray]]) -> None:
        """Replace all parameters (deep-copied from *params*)."""
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]

    def __repr__(self) -> str:
        return f"MLP(layer_sizes={self.layer_sizes})"
