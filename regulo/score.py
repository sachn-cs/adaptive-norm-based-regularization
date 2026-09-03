"""Evaluation metrics for regression and classification.

Each metric is a callable class implementing the :class:`Metric`
abstract interface.  All metrics accept and return scalar floats so
that they can be composed freely in cross-validation loops.

Edge-case conventions
---------------------
* **Empty arrays.**  Functions that compute means (MSE, MAE, RMSE)
  raise ``ZeroDivisionError`` or return ``NaN`` if passed empty
  inputs, consistent with ``numpy.mean``.
* **Perfect predictions (R-squared).**  When ``ss_tot == 0``
  (constant target), R-squared is ``1.0`` if the residuals are
  also zero, otherwise ``0.0``.
* **No true samples for a class (balanced accuracy).**  Classes
  present only in ``y_pred`` are silently ignored; if ``y_true`` is
  empty the return value is ``0.0``.
"""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Metric", "Mse", "Mae", "Rmse", "R2", "Balanced"]


class Metric(ABC):
    """Abstract base class for scalar evaluation metrics."""

    name: str = ""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Mse(Metric):
    """Mean squared error: ``(1/n) * sum((y_true - y_pred)**2)``.

    Computed over the total scalar element count.  Compatible with
    ``numpy.mean`` semantics for arbitrary-shape arrays.
    """

    name = "mse"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))


class Mae(Metric):
    """Mean absolute error: ``(1/n) * sum(|y_true - y_pred|)``."""

    name = "mae"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))


class Rmse(Metric):
    """Root mean squared error: ``sqrt(MSE)``.

    In the same units as the target.  Implemented via composition
    with :class:`Mse` rather than a separate formula so there is
    exactly one source of truth for the MSE computation.
    """

    name = "rmse"

    def __init__(self) -> None:
        self.mse = Mse()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(self.mse(y_true, y_pred)))


class R2(Metric):
    """Coefficient of determination.

    ``R^2 = 1 - SS_res / SS_tot`` where
    ``SS_res = sum((y - yhat)^2)`` and
    ``SS_tot = sum((y - mean(y))^2)``.

    Edge cases:
    * If ``SS_tot == 0`` and ``SS_res == 0``, returns ``1.0``.
    * If ``SS_tot == 0`` and ``SS_res > 0``, returns ``0.0``.
    """

    name = "r2"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot


class Balanced(Metric):
    """Balanced accuracy for multi-class classification.

    Unweighted mean of per-class recall (sensitivity).  Classes
    present only in ``y_pred`` are silently ignored.  Empty inputs
    return ``0.0``.
    """

    name = "balanced"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        classes = np.unique(y_true)
        per_class_acc = []
        for c in classes:
            mask = y_true == c
            if np.any(mask):
                per_class_acc.append(float(np.mean(y_pred[mask] == c)))
        return float(np.mean(per_class_acc)) if per_class_acc else 0.0
