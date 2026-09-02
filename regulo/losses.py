"""Loss functions with forward and backward methods.

Provides differentiable loss functions used during training.  Each loss
exposes a :meth:`forward` (scalar evaluation) and a :meth:`backward`
(analytical gradient w.r.t. the prediction).  The ``1 / n_samples``
averaging convention is applied *inside* the backward pass so that
gradients can be directly consumed by the optimizer.

Supported losses
----------------
* :class:`MSELoss` -- mean squared error for regression tasks.
* :class:`CrossEntropyLoss` -- softmax cross-entropy for multi-class
  classification with integer labels.

Gradient scaling convention
---------------------------
Gradients are **summed** across samples in the network backward pass
(``FullyConnectedNetwork.backward``) and **divided by n** in the loss
backward pass.  This matches standard deep-learning convention: the
optimizer receives ``(1/n) * sum_i nabla L_i``.

Numerical stability
-------------------
:class:`CrossEntropyLoss` subtracts the row-max logit before
exponentiation to avoid overflow, and clips log-probabilities with a
floor of ``1e-15`` to prevent ``log(0)``.
"""

import numpy as np


class MSELoss:
    r"""Mean squared error loss for regression.

    Computes the element-wise mean of squared residuals:

    .. math::

        L = \frac{1}{N} \sum_{i,j} (y_{ij} - \hat{y}_{ij})^2
            \quad\text{where } N = \texttt{y\_pred.size}

    The gradient w.r.t. ``y_pred`` is ``2 (y_pred - y_true) / N`` --
    the same ``N`` (the total scalar element count, not the number of
    rows) appears in both forward and backward, so the gradient is the
    exact derivative of the forward value.

    Why divide by ``N`` rather than by ``n_samples``
    ------------------------------------------------
    For ``y_pred`` of shape ``(n_samples, n_outputs)`` we set
    ``N = n_samples * n_outputs`` -- the total number of scalar
    prediction entries.  This matches :func:`numpy.mean` (which divides
    by ``size``) and the convention used by PyTorch's
    :code:`MSELoss(reduction='mean')`.  Without this convention,
    gradient magnitude would scale with the output dimension, which
    would require the user to compensate via the learning rate.

    When to use
    -----------
    Use for continuous-valued targets.  For classification tasks use
    :class:`CrossEntropyLoss` instead.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar MSE loss.

        Args:
            y_pred: Predictions of any shape broadcastable to ``y_true``,
                typically ``(n_samples, n_outputs)`` or ``(n_samples,)``.
            y_true: Targets with a shape broadcastable to ``y_pred``,
                typically ``(n_samples, n_outputs)`` or ``(n_samples,)``.

        Returns:
            Mean squared error as a Python ``float``.

        Notes:
            NumPy broadcasting applies to the subtraction, so inputs of
            compatible but unequal shapes (e.g. ``(n, 3)`` and ``(n,)``)
            are accepted silently rather than raising ``ValueError``.
        """
        diff = y_pred - y_true
        return float(np.mean(diff**2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of MSE w.r.t. ``y_pred``.

        The returned gradient is
        ``(2 / y_pred.size) * (y_pred - y_true)`` -- an array with the
        same shape as ``y_pred`` and already divided by ``N`` (total
        scalar elements).  It can be fed directly into
        :meth:`~regulo.network.FullyConnectedNetwork.backward` without
        further rescaling.

        Args:
            y_pred: Predictions, typically ``(n_samples, n_outputs)``.
            y_true: Targets with a shape broadcastable to ``y_pred``.

        Returns:
            Gradient array with the same shape as ``y_pred``.

        Notes:
            Because both the forward and backward operations divide by
            ``y_pred.size`` (rather than by ``y_pred.shape[0]``),
            gradient magnitude is invariant to the output dimension --
            a ``(100, 3)`` and a ``(100, 1)`` prediction of equal
            quality receive gradients of the same order of magnitude.
        """
        n_elements = y_pred.size
        return 2.0 * (y_pred - y_true) / n_elements


class CrossEntropyLoss:
    """Softmax cross-entropy loss for multi-class classification.

    Applies a numerically stable softmax internally and computes
    ``-log(p[y_true])`` averaged over samples.  Labels must be integer
    class indices in ``[0, n_classes)``.

    The softmax is embedded in the forward pass (not exposed separately)
    so that numerical stability optimizations can be applied.  The
    backward pass computes ``(softmax(logits) - one_hot(y_true)) / n``,
    which is the well-known closed-form gradient of softmax cross-entropy.

    When to use
    -----------
    Use for classification with integer labels.  For regression use
    :class:`MSELoss` instead.
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the scalar cross-entropy loss.

        Args:
            logits: Raw scores of shape ``(n_samples, n_classes)``.
                These are pre-softmax values; the softmax is applied
                internally for numerical stability.
            y_true: Integer class labels of shape ``(n_samples,)`` with
                values in ``[0, n_classes)``.

        Returns:
            Mean cross-entropy as a Python ``float``.

        Raises:
            IndexError: If ``y_true`` has fewer entries than ``logits``
                or if any label falls outside ``[0, n_classes)``.  NumPy's
                fancy-indexing surfaces the out-of-range label directly;
                this method performs no explicit validation, so the
                error originates inside the indexing expression.
        """
        # Numerically stable softmax: subtract the row-max so the
        # largest exponentiated logit is 1 and ``exp(.)`` never
        # overflows.  Subtracting a constant leaves softmax(z) invariant.
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        # Cross-entropy: -log(p[y_true]) with a floor for numerical
        # safety.  The 1e-15 floor prevents log(0) when the model is
        # extremely confident but wrong; without it we would emit
        # -inf and poison downstream gradients.
        log_probs = np.log(probs[np.arange(len(y_true)), y_true] + 1e-15)
        return float(-np.mean(log_probs))

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of cross-entropy w.r.t. ``logits``.

        The gradient equals ``(softmax(logits) - one_hot(y_true)) / n``
        where ``n = logits.shape[0]``.

        This is the standard closed-form result: the cross-entropy
        gradient w.r.t. logits simplifies to the difference between the
        predicted probability distribution and the one-hot target.

        Args:
            logits: Raw scores of shape ``(n_samples, n_classes)``.
            y_true: Integer class labels of shape ``(n_samples,)``.

        Returns:
            Gradient array with the same shape as ``logits``.
        """
        # Recompute softmax (no caching across forward/backward).
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        n_samples = logits.shape[0]
        # Subtract 1.0 at the true class position: softmax - one_hot.
        grad = probs.copy()
        grad[np.arange(len(y_true)), y_true] -= 1.0
        return grad / n_samples
