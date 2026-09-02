"""Training loop with mini-batching, early stopping, and validation.

The :class:`Trainer` class orchestrates the epoch-level training loop,
including:

* Mini-batch sampling with per-epoch shuffling.
* Forward pass, loss computation, and regularizer penalty.
* Back-propagation and parameter update via :class:`~regulo.optimizer.Adam`.
* Optional validation loss tracking and early stopping.
* Verbose progress printing.

Covridge / Sparridge layer handling
------------------------------------
:class:`~regulo.regularizers.Covridge` and :class:`~regulo.regularizers.Sparridge`
are applied **only to the first weight matrix** because the empirical Gram
matrix ``C_{delta,n}`` is defined over the input dimension and therefore
only matches ``W^{(1)}``.  This is a design assumption: the paper defines
``C_{delta,n} = (1/n) X^T X + delta I_p`` where ``X`` is the input
matrix, so applying it to deeper layers would require computing a
different Gram matrix for each layer's activations.

All other regularizers (Ridge, Lasso, ElasticNet, NoRegularizer) are
applied to **every** weight matrix, which is the standard approach.

Gradient flow
-------------
Each update step computes:

1. ``y_pred = network.forward(x_batch)``
2. ``loss = loss_fn.forward(y_pred, y_batch)`` (plus any regularizer penalty)
3. ``dloss = loss_fn.backward(y_pred, y_batch)`` -- already divided by
   ``n`` (sample count for cross-entropy, total elements for MSE).
4. ``grads = network.backward(dloss)`` -- raw parameter gradients.
5. ``grads["weights"][i] += regularizer.gradient(weights[i])`` for every
   layer, or only for the first when the regularizer is geometry-aware.
6. ``new_params = optimizer.step(params, grads)`` -- Adam update.

The ``1/N`` scaling lives in the loss backward (not the network
backward), so the network backward always operates on per-batch
sum-of-gradients and the same scaling convention is applied at every
call site.

Lifecycle
---------
A :class:`Trainer` is intended for a single training run on a single
network.  Re-using the same trainer for a second independent
optimization (different data, different network) is **not** supported:
the network weights, optimizer state, and history are all carried over.
Construct a fresh :class:`Trainer` for each run.
"""

from typing import Dict, List, Optional

import numpy as np

import regulo.losses as losses
from regulo.network import FullyConnectedNetwork
from regulo.optimizer import Adam
from regulo.regularizers import Covridge, Regularizer, Sparridge


class Trainer:
    """End-to-end trainer for the manual NumPy network.

    Ties together a :class:`~regulo.network.FullyConnectedNetwork`, a loss
    function, a regularizer, and an :class:`~regulo.optimizer.Adam`
    optimizer into a single :meth:`fit` / :meth:`predict` interface.

    The trainer is **not** reusable for multiple independent runs without
    calling :meth:`predict` (which does not reset state) -- create a new
    ``Trainer`` instance for each independent experiment.

    Attributes:
        history: Dictionary with ``"train_loss"`` and ``"val_loss"`` lists
            populated during :meth:`fit``.

    Thread safety
    -------------
    The trainer is **not** thread-safe: it mutates the network, optimizer,
            and history during :meth:`fit`.
    """

    def __init__(
        self,
        network: FullyConnectedNetwork,
        loss_fn: losses.MSELoss | losses.CrossEntropyLoss,
        regularizer: Regularizer,
        optimizer: Adam,
        batch_size: int = 32,
        epochs: int = 500,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialise the trainer with all component objects.

        Args:
            network: The network to train.  Will be mutated in place
                during :meth:`fit`.
            loss_fn: Loss function with ``forward`` / ``backward`` methods.
            regularizer: Regularizer applied to weight matrices.
            optimizer: Adam optimizer instance.
            batch_size: Mini-batch size (default ``32``).  The last batch
                may be smaller if ``n_samples`` is not divisible by
                ``batch_size``.
            epochs: Maximum number of training epochs (default ``500``).
            early_stopping: Enable early stopping on validation loss.
                Requires ``x_val`` and ``y_val`` to be passed to
                :meth:`fit`.
            patience: Number of epochs without improvement before
                stopping (default ``10``).
            verbose: Print progress every 50 epochs.
        """
        self.network = network
        self.loss_fn = loss_fn
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the network on *x_train* with optional validation.

        Each epoch randomly permutes ``x_train`` and iterates over
        mini-batches.  For every batch the trainer:

        1. Runs the network forward pass.
        2. Computes the data loss and (per layer) the regularizer penalty.
        3. Runs the loss backward and then the network backward to
           obtain parameter gradients.
        4. Adds the analytic regularizer gradient to every weight
           gradient (geometry-aware regularizers modify only
           ``weights[0]``; see the module docstring).
        5. Applies one :meth:`~regulo.optimizer.Adam.step`.

        Input shape flexibility
        -----------------------
        ``y_train`` is accepted as any of the common shapes
        (``(n, 1)``, ``(n,)``, or higher-dimensional ``(n, k)``); the
        loss function decides how the dimensions are interpreted, and
        the network's last layer determines its column count.

        Early stopping
        --------------
        When ``early_stopping=True`` and ``x_val``/``y_val`` are
        provided, the best validation-loss snapshot is restored at the
        end of training (also when the loop exits via the patience
        counter).

        Side effects
        ------------
        * Mutates the network weights and biases in place.
        * Increments the optimizer step counter and updates its moment
          buffers.
        * Appends to :attr:`history["train_loss"]` once per epoch;
          appends to :attr:`history["val_loss"]` only when validation
          data is supplied.

        Complexity
        ----------
        ``O(epochs * n_train * (F + B + U))`` where ``F``, ``B``, ``U``
        are the costs of forward, backward, and optimizer step on a
        single batch.  Memory use is proportional to ``batch_size *
        depth`` (caches for the backward pass).

        Args:
            x_train: Training inputs of shape ``(n_train, p)``.
            y_train: Training targets, shape ``(n_train,)`` for
                classification or ``(n_train, n_outputs)`` for
                regression / multi-output classification.
            x_val: Validation inputs (optional).  Required for early
                stopping and for populating ``history['val_loss']``.
            y_val: Validation targets (optional).  Required for early
                stopping and for populating ``history['val_loss']``.

        Raises:
            ValueError: If ``early_stopping`` is ``True`` but ``x_val``
                or ``y_val`` is ``None``.
        """
        if self.early_stopping and (x_val is None or y_val is None):
            raise ValueError(
                "early_stopping requires x_val and y_val to be provided."
            )
        n_samples = x_train.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0
        best_params: Optional[Dict[str, List[np.ndarray]]] = None

        for epoch in range(self.epochs):
            # Shuffle indices each epoch for stochasticity.
            indices = np.random.permutation(n_samples)
            epoch_losses: List[float] = []
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass and data loss.
                y_pred = self.network.forward(x_batch)
                loss = self.loss_fn.forward(y_pred, y_batch)

                # Regularization penalty.  Geometry-aware regularizers
                # (Covridge, Sparridge) are applied only to the first
                # weight matrix because C_{delta,n} is computed from
                # the input dimension -- see the module docstring.
                reg_penalty = 0.0
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    reg_penalty += self.regularizer.penalty(
                        self.network.weights[0]
                    )
                else:
                    for w in self.network.weights:
                        reg_penalty += self.regularizer.penalty(w)
                total_loss = loss + reg_penalty
                epoch_losses.append(total_loss)

                # Backward pass through loss and network.
                dloss = self.loss_fn.backward(y_pred, y_batch)
                grads = self.network.backward(dloss)

                # Accumulate analytic regularizer gradients onto the
                # weight gradients.  Mirrors the penalty branch above.
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    grads["weights"][0] += self.regularizer.gradient(
                        self.network.weights[0]
                    )
                else:
                    for i, w in enumerate(self.network.weights):
                        grads["weights"][i] += self.regularizer.gradient(w)

                # Optimizer step: hand the network's current parameters
                # along with their (regularized) gradients to Adam and
                # replace the parameters with the returned updates.
                params = {
                    "weights": self.network.weights,
                    "biases": self.network.biases,
                }
                new_params = self.optimizer.step(params, grads)
                self.network.set_params(new_params)

            avg_train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(avg_train_loss)

            # Validation evaluation.
            if x_val is not None and y_val is not None:
                val_pred = self.network.forward(x_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                if isinstance(self.regularizer, (Covridge, Sparridge)):
                    val_loss += self.regularizer.penalty(
                        self.network.weights[0]
                    )
                else:
                    for w in self.network.weights:
                        val_loss += self.regularizer.penalty(w)
                self.history["val_loss"].append(float(val_loss))

                # Early stopping logic.
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Deep-copy via FullyConnectedNetwork.get_params
                        # so subsequent weight updates don't mutate
                        # the snapshot.
                        best_params = self.network.get_params()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            if best_params is not None:
                                self.network.set_params(best_params)
                            break

            if self.verbose and (epoch + 1) % 50 == 0:
                msg = (
                    f"Epoch {epoch + 1}/{self.epochs}"
                    f" -- train loss: {avg_train_loss:.4f}"
                )
                if x_val is not None:
                    msg += f", val loss: {self.history['val_loss'][-1]:.4f}"
                print(msg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input.

        Runs a single forward pass through the underlying network.  Does
        **not** mutate the network weights or the optimizer state.
        Suitable for evaluation, calibration, or ensembling without
        affecting the trained model.

        Args:
            x: Input array of shape ``(n_samples, n_features)``.

        Returns:
            Network output of shape ``(n_samples, n_outputs)``.  For
            regression these are continuous predictions; for
            classification these are pre-softmax logits -- pass the
            output through ``np.argmax(..., axis=1)`` to recover class
            labels.

        Notes:
            For ``n_samples == 1`` the forward pass still returns a
            ``(1, n_outputs)`` 2-D array.  Subsequent NumPy operations
            (``np.squeeze``, ``item()``) may be needed to reduce it to
            a scalar.
        """
        return self.network.forward(x)
