"""Evaluation metrics for regression and classification.

Pure-NumPy implementations with no runtime dependency on sklearn.
All functions accept and return scalar floats so that they can be
composed freely in cross-validation loops.

Edge-case conventions
---------------------
* **Empty arrays.** Functions that compute means (MSE, MAE, RMSE) will
  raise ``ZeroDivisionError`` or return ``NaN`` if passed empty inputs,
  consistent with ``numpy.mean``.
* **Perfect predictions (R-squared).** When ``ss_tot == 0`` (constant
  target), R-squared is ``1.0`` if the residuals are also zero, otherwise
  ``0.0``.  This avoids division by zero while remaining mathematically
  defensible: a constant model achieves R-squared = 0, and a perfect fit
  achieves R-squared = 1.
* **No true samples for a class (balanced accuracy).** Classes present
  only in ``y_pred`` are silently ignored; if ``y_true`` is empty the
  return value is ``0.0``.
"""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between ``y_true`` and ``y_pred``.

    Computes ``(1/n) * sum((y_true - y_pred)^2)`` over the total
    number of scalar entries (``n = y_true.size``), which is the
    standard ``numpy.mean`` convention -- the subtraction and squaring
    broadcast across any leading dimensions, so ``y_true`` and
    ``y_pred`` may have any compatible shape including ``(n,)``,
    ``(n, k)``, or higher.

    Args:
        y_true: Ground-truth values (any shape).
        y_pred: Predicted values with a shape broadcastable to
            *y_true*.

    Returns:
        Scalar MSE as a Python ``float``.

    Notes:
        MSE penalizes large errors disproportionately due to the
        squaring.  When outliers matter, prefer MAE
        (:func:`mean_absolute_error`).
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between ``y_true`` and ``y_pred``.

    Computes ``(1/n) * sum(|y_true - y_pred|)``.  More robust to
    outliers than :func:`mean_squared_error` because errors are
    penalized linearly rather than quadratically.

    Args:
        y_true: Ground-truth values (any shape).
        y_pred: Predicted values with a shape broadcastable to
            *y_true*.

    Returns:
        Scalar MAE as a Python ``float``.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (square root of MSE).

    RMSE is in the same units as the target, making it more
    interpretable than MSE.  It penalizes large errors more heavily
    than MAE due to the squaring.

    Args:
        y_true: Ground-truth values (any shape broadcastable to
            *y_pred*).
        y_pred: Predicted values with a shape broadcastable to
            *y_true*.

    Returns:
        Scalar RMSE as a Python ``float``.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R-squared).

    ``R^2 = 1 - SS_res / SS_tot`` where ``SS_res = sum((y - yhat)^2)``
    and ``SS_tot = sum((y - mean(y))^2)``.

    Interpretation:

    * ``R^2 = 1`` -- perfect fit.
    * ``R^2 = 0`` -- model performs no better than predicting the mean.
    * ``R^2 < 0`` -- model performs worse than predicting the mean.

    Edge cases:

    * If ``SS_tot == 0`` (constant target) and ``SS_res == 0`` (perfect
      fit), returns ``1.0``.
    * If ``SS_tot == 0`` and ``SS_res > 0``, returns ``0.0``.

    Args:
        y_true: Ground-truth values (any shape broadcastable to
            *y_pred*).
        y_pred: Predicted values with a shape broadcastable to
            *y_true*.

    Returns:
        Scalar R-squared in ``(-inf, 1]`` as a Python ``float``.

    Notes:
        This implementation computes :code:`SS_tot` using the mean
        over the **flattened** array (``numpy.mean(y_true)``), which
        is the standard convention in scikit-learn's
        :code:`r2_score` for 1-D targets.  For multi-output arrays the
        result is the flattened analog of sklearn's
        :code:`r2_score(..., multioutput='uniform_average')`.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Balanced accuracy for multi-class classification.

    Computes the unweighted mean of per-class recall (sensitivity):

    .. math::

        \text{BA} = \frac{1}{|C|} \sum_{c \in C}
            \frac{1}{|\{i : y_i = c\}|} \; |\{i : y_i = c, \hat{y}_i = c\}|

    where ``C`` is the set of unique labels in ``y_true``.  Classes
    present only in ``y_pred`` are silently ignored.

    Unlike plain accuracy, balanced accuracy is not inflated by class
    imbalance: a model that always predicts the majority class will
    have balanced accuracy equal to that class's prior fraction, not
    ``1.0``.

    Edge cases:

    * Empty inputs return ``0.0``.
    * A class with no supporting samples in ``y_true`` is skipped
      (this cannot happen by construction when classes are derived
      from ``y_true``).

    Args:
        y_true: True integer labels of shape ``(n_samples,)``.
        y_pred: Predicted integer labels of shape ``(n_samples,)``.

    Returns:
        Balanced accuracy in ``[0, 1]`` as a Python ``float``.
    """
    classes = np.unique(y_true)
    per_class_acc = []
    for c in classes:
        # ``np.any(mask)`` is always True here because ``c`` comes
        # from ``np.unique(y_true)``; the guard exists for defensive
        # clarity, not behavioural necessity.
        mask = y_true == c
        if np.any(mask):
            acc = np.mean(y_pred[mask] == c)
            per_class_acc.append(acc)
    return float(np.mean(per_class_acc)) if per_class_acc else 0.0
