"""Training loop with mini-batching, early stopping, and validation.

The :class:`Runner` class orchestrates the epoch-level training
loop:

* Mini-batch sampling with per-epoch shuffling.
* Forward pass, loss computation, and penalty value.
* Back-propagation and parameter update via :class:`regulo.adam.Adam`.
* Optional validation loss tracking and early stopping.

Covridge / Sparridge layer handling
------------------------------------
:class:`Covridge` and :class:`Sparridge` are applied only to the
first weight matrix because the empirical Gram matrix
``C_{delta,n}`` is defined over the input dimension and therefore
only matches ``W^{(1)}``.  The dispatch uses the polymorphic
:meth:`Penalty.applies` hook -- no ``isinstance`` checks.

Gradient flow
-------------
Each update step computes:

1. ``y_pred = mlp(x_batch)``
2. ``loss = loss_fn.value(y_pred, y_batch)``
3. ``dloss = loss_fn.grad(y_pred, y_batch)``
4. ``grads = mlp.grad(dloss)``
5. ``grads["weights"][i] += penalty.grad(weight_i, i)`` for layers
   where ``penalty.applies(i)``.
6. ``new_params = adam.step(params, grads)``

Lifecycle
---------
A :class:`Runner` instance is single-use by design: it owns the
network weights, optimizer state, and history.  To re-train from
scratch, call :meth:`reset` or construct a fresh :class:`Runner`.
"""

from typing import Dict, List, Optional

import numpy as np

from regulo.adam import Adam
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import Penalty

__all__ = ["Runner"]


class Runner:
    """End-to-end trainer for the manual NumPy network.

    Ties together a :class:`MLP`, a :class:`Loss`, a :class:`Penalty`,
    and an :class:`Adam` optimizer into a single
    :meth:`fit` / :meth:`predict` interface.

    Attributes:
        task: ``"regression"`` or ``"classification"``, derived from
            the loss type.
        history: Dictionary with ``"train"`` and ``"val"`` lists
            populated during :meth:`fit`.
    """

    def __init__(
        self,
        mlp: MLP,
        loss_fn: Loss,
        penalty: Penalty,
        adam: Adam,
        batch_size: int = 32,
        epochs: int = 500,
        early_stopping: bool = False,
        patience: int = 10,
    ) -> None:
        """Initialise the runner with all component objects."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        self.mlp = mlp
        self.loss_fn = loss_fn
        self.penalty = penalty
        self.adam = adam
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.task = "classification" if loss_fn.name == "softmax" else "regression"
        self.history: Dict[str, List[float]] = {"train": [], "val": []}

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Train the network on *x_train* with optional validation.

        Args:
            x_train: Training inputs of shape ``(n_train, p)``.
            y_train: Training targets.
            x_val: Validation inputs (optional).  Required for
                early stopping.
            y_val: Validation targets (optional).
            seed: Optional integer seed for mini-batch shuffling.
        """
        if self.early_stopping and (x_val is None or y_val is None):
            raise ValueError(
                "early_stopping requires x_val and y_val to be provided."
            )
        n = x_train.shape[0]
        rng = np.random.default_rng(seed)
        best_val = float("inf")
        patience_left = self.patience
        best_state: Optional[Dict[str, List[np.ndarray]]] = None

        for epoch in range(self.epochs):
            indices = rng.permutation(n)
            losses: List[float] = []
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]

                y_pred = self.mlp(x_batch)
                loss = self.loss_fn.value(y_pred, y_batch)

                # Penalty value and gradient over applicable layers.
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        loss += self.penalty.value(w, i)
                losses.append(loss)

                dloss = self.loss_fn.grad(y_pred, y_batch)
                grads = self.mlp.grad(dloss)
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        grads["weights"][i] = grads["weights"][i] + self.penalty.grad(
                            w, i
                        )

                params = {
                    "weights": self.mlp.weights,
                    "biases": self.mlp.biases,
                }
                new_params = self.adam.step(params, grads)
                # NaN / Inf guard: detect non-finite parameters or
                # gradients and abort with a clear error pointing to
                # the offending epoch and batch index.
                for i, w in enumerate(new_params["weights"]):
                    if not np.all(np.isfinite(w)):
                        raise FloatingPointError(
                            f"non-finite weight at epoch {epoch + 1}, "
                            f"batch {start // self.batch_size}, layer {i}"
                        )
                for i, b in enumerate(new_params["biases"]):
                    if not np.all(np.isfinite(b)):
                        raise FloatingPointError(
                            f"non-finite bias at epoch {epoch + 1}, "
                            f"batch {start // self.batch_size}, layer {i}"
                        )
                self.mlp.load(new_params)

            self.history["train"].append(float(np.mean(losses)))

            if x_val is not None and y_val is not None:
                val_pred = self.mlp(x_val)
                val_loss = self.loss_fn.value(val_pred, y_val)
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        val_loss += self.penalty.value(w, i)
                self.history["val"].append(float(val_loss))

                if self.early_stopping:
                    if val_loss < best_val:
                        best_val = val_loss
                        patience_left = self.patience
                        best_state = self.mlp.state()
                    else:
                        patience_left -= 1
                        if patience_left < 0:
                            if best_state is not None:
                                self.mlp.load(best_state)
                            break

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate raw predictions (logits for classification)."""
        return self.mlp(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Generate softmax probabilities (classification only)."""
        if self.task != "classification":
            raise NotImplementedError(
                "predict_proba is only defined for classification "
                "(Softmax loss) runners."
            )
        logits = self.mlp(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        """Generate class index predictions (classification only)."""
        return np.argmax(self.predict_proba(x), axis=1)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the network weights and optimizer state.

        Re-initialises :attr:`mlp` from scratch (with *seed*) and
        clears the optimizer state and training history.
        """
        layer_sizes = self.mlp.layer_sizes
        self.mlp = MLP(layer_sizes, seed=seed)
        self.adam.reset()
        self.history = {"train": [], "val": []}

    def warm_start(self, path: str) -> None:
        """Load weights only from a directory produced by :func:`regulo.store.save`.

        Validates the architecture recorded in ``meta.json`` against
        :attr:`mlp.layer_sizes` and replaces the current weights
        without disturbing optimizer state.
        """
        from pathlib import Path

        from regulo.store import load_meta

        meta = load_meta(path)
        if list(meta["layer_sizes"]) != list(self.mlp.layer_sizes):
            raise ValueError(
                f"Architecture mismatch: on-disk {meta['layer_sizes']} "
                f"!= in-memory {self.mlp.layer_sizes}."
            )
        p = Path(path)
        weights = np.load(p / "weights.npz")
        biases = np.load(p / "biases.npz")
        for i, w in enumerate(self.mlp.weights):
            w[...] = weights[f"w{i}"]
        for i, b in enumerate(self.mlp.biases):
            b[...] = biases[f"b{i}"]

    def __repr__(self) -> str:
        return (
            f"Runner(mlp={self.mlp!r}, loss={self.loss_fn!r}, "
            f"penalty={self.penalty!r}, adam={self.adam!r})"
        )
