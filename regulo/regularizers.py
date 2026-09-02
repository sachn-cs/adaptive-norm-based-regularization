"""Regularizer penalties and analytical gradients.

This module defines the regularizers used throughout the ANBR
reproduction.  Each regularizer is a stateless callable object exposing
two methods:

* :meth:`Regularizer.penalty` -- scalar penalty ``Omega(W)`` added to the
  empirical loss.
* :meth:`Regularizer.gradient` -- analytical gradient ``nabla_W Omega(W)``
  of the penalty with respect to the weight matrix, used by
  :class:`~regulo.trainer.Trainer` during backpropagation.

The concrete penalties map directly to equations in Qasim & Javed:

==========================  ============================================
Class                       Paper equation
==========================  ============================================
:class:`Ridge`              ``lambda ||W||_F^2``
:class:`Lasso`              ``gamma ||W||_1``
:class:`ElasticNet`         ``alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``
:class:`Covridge`           ``lambda1 ||C_{delta,n}^{1/2} W||_F^2 + lambda2 ||W||_F^2``
:class:`Sparridge`          ``lambda1 ||C_{delta,n}^{1/2} W||_F^2 + gamma ||W||_1``
==========================  ============================================

Design pattern
--------------
All regularizers follow the Strategy pattern: they implement the same
``penalty`` / ``gradient`` interface so that :class:`~regulo.trainer.Trainer`
can dispatch them without type checks for the common case (Ridge, Lasso,
ElasticNet, NoRegularizer).  Covridge and Sparridge require a special
branch in the trainer because the Gram matrix ``C_{delta,n}`` is defined
over the input dimension and therefore only matches the first weight
matrix.

Numerical stability
-------------------
:class:`Covridge` and :class:`Sparridge` factor ``C_{delta,n}`` once at
construction using a symmetric eigendecomposition (``numpy.linalg.eigh``).
The decomposition is symmetric by construction (``C_n`` and ``I`` are
symmetric and so is their sum), which is why ``eigh`` is preferred over
``scipy.linalg.sqrtm``: ``eigh`` returns real eigenvalues and avoids the
complex round-trip that ``sqrtm`` performs.  Tiny negative eigenvalues
introduced by floating-point error are clamped to zero before the
square root is taken, guaranteeing a real PSD square root.

Complexity
----------
All penalty/gradient evaluations are linear in the number of weight
entries -- O(m * n) for a weight matrix of shape ``(m, n)``.  The
eigendecomposition performed once per ``Covridge``/``Sparridge``
instance is O(p^3) where ``p`` is the input dimension.  After
construction, the per-step cost is identical to that of Ridge.

Thread safety
-------------
Instances are immutable after construction (hyperparameters and cached
matrices are set once and only read), so they are safe to share across
threads provided the numpy arrays are not mutated by callers.
"""

from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    """Abstract base class for weight-matrix regularizers.

    All regularizers share the same interface so that
    :class:`~regulo.trainer.Trainer` can dispatch them uniformly.  Subclasses
    must implement :meth:`penalty` and :meth:`gradient`; both are pure
    functions of the weight matrix and have no side effects.

    Thread safety
    -------------
    Instances are immutable after construction (hyperparameters are
    stored once and only read), so they are safe to share across
    threads provided the underlying numpy arrays are not mutated by
    callers.  The network weights passed to :meth:`penalty` and
    :meth:`gradient` are never copied.
    """

    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        """Compute the scalar regularization penalty.

        Args:
            weights: Weight matrix of shape ``(in_features, out_features)``.

        Returns:
            Scalar penalty value ``Omega(W)`` to be added to the empirical
            loss.

        Notes:
            This is a pure function with no side effects; it does not
            modify ``weights``.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute the gradient of the penalty w.r.t. ``weights``.

        Args:
            weights: Weight matrix of shape ``(in_features, out_features)``.

        Returns:
            Gradient array with the same shape as ``weights``.  The caller
            (``Trainer``) is responsible for accumulating this with the
            data-gradient from backpropagation before invoking the
            optimizer.

        Notes:
            This is a pure function with no side effects; it does not
            modify ``weights``.
        """
        raise NotImplementedError


class NoRegularizer(Regularizer):
    """No-op regularizer used as the unregularized baseline.

    Always returns zero penalty and zero gradient so that
    :class:`~regulo.trainer.Trainer` can iterate over regularizer instances
    without special-casing the absence of regularization.  This
    eliminates the need for ``if reg is None`` branches in the training
    loop.
    """

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``0.0``.

        Args:
            weights: Ignored.  Present to satisfy the abstract interface.

        Returns:
            ``0.0`` regardless of input.
        """
        return 0.0

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return a zero array shaped like ``weights``.

        Args:
            weights: Reference shape used to construct the result.

        Returns:
            Array of zeros with the same shape and dtype as ``weights``.
        """
        return np.zeros_like(weights)


class Ridge(Regularizer):
    r"""Ridge (L2 / Tikhonov) penalty: ``lambda ||W||_F^2``.

    Penalty: ``Omega(W) = lambda * sum_{i,j} w_{ij}^2``.
    Gradient: ``nabla_W Omega(W) = 2 lambda W``.

    Equivalent to a Gaussian prior on the weights.  Smooth and strongly
    convex in ``W``; encourages small magnitudes without inducing
    sparsity.  Setting ``lambda_ = 0`` reduces this to
    :class:`NoRegularizer` (the penalty and gradient are still computed
    numerically rather than short-circuited).

    References
    ----------
    Hoerl, A. E. & Kennard, R. W. (1970).  "Ridge Regression: Biased
    Estimation for Nonorthogonal Problems."  *Technometrics*, 12(1), 55--67.
    """

    def __init__(self, lambda_: float) -> None:
        """Store the non-negative regularization strength.

        Args:
            lambda_: Penalty weight ``lambda``.  Must be ``>= 0``; negative
                values are mathematically meaningless and rejected.

        Raises:
            ValueError: If ``lambda_ < 0``.
        """
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        self.lambda_ = lambda_

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``lambda * ||W||_F^2``.

        Args:
            weights: Weight matrix of shape ``(in_features, out_features)``.

        Returns:
            Scalar Frobenius-norm-squared penalty.
        """
        return self.lambda_ * float(np.sum(weights**2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return ``2 lambda W``.

        Args:
            weights: Weight matrix.

        Returns:
            Gradient array, same shape as ``weights``.
        """
        return 2.0 * self.lambda_ * weights


class Lasso(Regularizer):
    r"""Lasso (L1) penalty: ``gamma ||W||_1``.

    Penalty: ``Omega(W) = gamma * sum_{i,j} |w_{ij}|``.
    Subgradient: ``nabla_W Omega(W) = gamma * sign(W)`` with
    ``sign(0) := 0``.

    Promotes sparsity by driving individual weights exactly to zero.
    Non-smooth at the origin, so the implementation uses a subgradient
    rather than a true gradient.  The subgradient is well-defined at
    ``W = 0`` because ``numpy.sign(0) == 0``.

    References
    ----------
    Tibshirani, R. (1996).  "Regression Shrinkage and Selection via
    the Lasso."  *Journal of the Royal Statistical Society: Series B*,
    58(1), 267--288.
    """

    def __init__(self, gamma: float) -> None:
        """Store the non-negative sparsity weight.

        Args:
            gamma: Penalty weight ``gamma``.  Must be ``>= 0``.

        Raises:
            ValueError: If ``gamma < 0``.
        """
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = gamma

    def penalty(self, weights: np.ndarray) -> float:
        """Return ``gamma * ||W||_1``.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar L1 penalty.
        """
        return self.gamma * float(np.sum(np.abs(weights)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return the subgradient ``gamma * sign(W)``.

        ``numpy.sign`` already returns ``0`` for zero inputs, so the
        subgradient is well-defined at ``W = 0`` and requires no
        additional clipping.

        Args:
            weights: Weight matrix.

        Returns:
            Subgradient array, same shape as ``weights``.
        """
        return self.gamma * np.sign(weights)


class ElasticNet(Regularizer):
    r"""Elastic Net: ``alpha gamma ||W||_1 + (1 - alpha)/2 * ||W||_F^2``.

    Convex combination of :class:`Lasso` (sparsity) and
    :class:`Ridge` (shrinkage).  At the endpoints it degenerates to:

    * ``alpha = 0`` -> ``(1/2) ||W||_F^2`` -- equivalent to
      :class:`Ridge` with ``lambda_ = 0.5``.
    * ``alpha = 1`` -> ``gamma ||W||_1`` -- equivalent to :class:`Lasso`
      with the same ``gamma``.

    Penalty: ``Omega(W) = alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``.
    Gradient: ``nabla_W Omega(W) = alpha gamma sign(W) + (1 - alpha) W``.

    References
    ----------
    Zou, H. & Hastie, T. (2005).  "Regularization and Variable Selection
    via the Elastic Net."  *Journal of the Royal Statistical Society:
    Series B*, 67(2), 301--320.
    """

    def __init__(self, alpha: float, gamma: float) -> None:
        """Store mixing parameter and penalty weight.

        Args:
            alpha: Mixing parameter in ``[0, 1]``.  ``0`` reduces to a
                pure quadratic penalty, ``1`` reduces to a pure L1
                penalty.
            gamma: Penalty weight ``gamma``.  Must be ``>= 0``.

        Raises:
            ValueError: If ``alpha`` not in ``[0, 1]`` or ``gamma < 0``.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma

    def penalty(self, weights: np.ndarray) -> float:
        """Return the elastic-net penalty.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar ``alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``.
        """
        l1 = self.alpha * self.gamma * float(np.sum(np.abs(weights)))
        l2 = (1.0 - self.alpha) * 0.5 * float(np.sum(weights**2))
        return l1 + l2

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        r"""Return ``alpha gamma sign(W) + (1 - alpha) W``.

        Args:
            weights: Weight matrix.

        Returns:
            Gradient array, same shape as ``weights``.
        """
        l1_grad = self.alpha * self.gamma * np.sign(weights)
        l2_grad = (1.0 - self.alpha) * weights
        return l1_grad + l2_grad


class Covridge(Regularizer):
    r"""Covridge penalty: ``lambda1 ||C_{delta,n}^{1/2} W||_F^2 + lambda2 ||W||_F^2``.

    The first term shrinks weights along the eigenvectors of the
    empirical Gram matrix ``C_n = (1/n) X^T X`` (stabilized by
    ``delta I_p`` to form ``C_{delta,n}``); the second term is a standard
    isotropic ridge.  Together they implement the geometry-aware
    shrinkage of the paper.

    Penalty: ``Omega(W) = lambda1 ||C_{delta,n}^{1/2} W||_F^2 + lambda2 ||W||_F^2``.
    Gradient: ``nabla_W Omega(W) = 2 lambda1 C_{delta,n} W + 2 lambda2 W``.

    Special cases
    -------------
    * ``C_{delta,n} = I`` reduces Covridge to :class:`Ridge` with
      ``lambda_ = lambda1 + lambda2``.
    * ``lambda2 = 0`` keeps only the geometry-aware term; ``lambda1 = 0``
      keeps only the isotropic term.

    Construction cost
    -----------------
    A single symmetric eigendecomposition of the ``p x p`` matrix is
    performed once and cached as ``_c_sqrt``.  The per-step penalty and
    gradient then cost O(p^2 q) respectively, where ``q`` is the output
    dimension of the layer being regularized.

    References
    ----------
    Qasim, M. & Javed, F.  "Adaptive Norm-Based Regularization for
    Neural Networks."  Equations 3.2.1 in ``docs/math.md``.
    """

    def __init__(
        self,
        lambda1: float,
        lambda2: float,
        c_delta_n: np.ndarray,
    ) -> None:
        """Store penalties and precompute ``C_{delta,n}^{1/2}``.

        The matrix square root is computed via symmetric eigendecomposition
        (``numpy.linalg.eigh``) and cached.  This is done once at
        construction time so that per-step penalty and gradient
        evaluations are O(p^2 q) rather than O(p^3).

        Args:
            lambda1: Geometry-aware penalty weight ``lambda1``.  Must be
                ``>= 0``.
            lambda2: Isotropic penalty weight ``lambda2``.  Must be
                ``>= 0``.
            c_delta_n: Stabilized Gram matrix of shape ``(p, p)``,
                typically ``(1/n) X^T X + delta I_p``.

        Raises:
            ValueError: If ``lambda1`` or ``lambda2`` is negative.

        Notes:
            The eigendecomposition uses ``numpy.linalg.eigh``, which is
            both faster and more numerically stable than
            ``scipy.linalg.sqrtm`` because ``C_{delta,n}`` is symmetric
            positive-semidefinite by construction.
        """
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be non-negative.")
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # Compute the matrix square root via symmetric eigendecomposition.
        # eigh is preferred over sqrtm because C_{delta,n} is symmetric by
        # construction, which makes the eigenbasis real and avoids the
        # complex-arithmetic detour of sqrtm.
        eigvals, eigvecs = np.linalg.eigh(c_delta_n)
        # Guard against tiny negative eigenvalues introduced by floating-
        # point error; clipping to 0 keeps the square root real-valued
        # without materially affecting the penalty for non-degenerate
        # C_{delta,n}.
        eigvals = np.maximum(eigvals, 0.0)
        self._c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def penalty(self, weights: np.ndarray) -> float:
        r"""Return ``lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2``.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar penalty.
        """
        cw = self._c_sqrt @ weights
        return self.lambda1 * float(np.sum(cw**2)) + self.lambda2 * float(
            np.sum(weights**2)
        )

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        r"""Return ``2 lambda1 C_{delta,n} W + 2 lambda2 W``.

        The first term is derived as
        ``nabla_W ||C^{1/2} W||_F^2 = 2 C^{1/2} (C^{1/2} W) = 2 C W``.
        The cached ``_c_sqrt`` is multiplied with itself rather than
        storing ``c_delta_n`` separately so that the gradient operates
        on the same matrix that defines the penalty (including any
        numerical clipping performed at construction time).

        Args:
            weights: Weight matrix.

        Returns:
            Gradient array, same shape as ``weights``.
        """
        c_mat = self._c_sqrt @ self._c_sqrt
        return (
            2.0 * self.lambda1 * (c_mat @ weights)
            + 2.0 * self.lambda2 * weights
        )


class Sparridge(Regularizer):
    r"""Sparridge penalty: ``lambda1 ||C_{delta,n}^{1/2} W||_F^2 + gamma ||W||_1``.

    Combines the geometry-aware shrinkage of :class:`Covridge` with
    the sparsity-inducing L1 term of :class:`Lasso`.  Useful when the
    data exhibit both correlated predictors and a sparse subset of
    informative features.

    Penalty: ``Omega(W) = lambda1 ||C_{delta,n}^{1/2} W||_F^2 + gamma ||W||_1``.
    Subgradient: ``nabla_W Omega(W) = 2 lambda1 C_{delta,n} W + gamma sign(W)``.

    Special cases
    -------------
    * ``C_{delta,n} = I`` reduces Sparridge to an elastic-net-like
      penalty with ``lambda1`` playing the role of the ridge weight.
    * ``gamma = 0`` removes the sparsity term and recovers the
      geometry-aware ridge of :class:`Covridge`.

    References
    ----------
    Qasim, M. & Javed, F.  "Adaptive Norm-Based Regularization for
    Neural Networks."  Equations 3.2.2 in ``docs/math.md``.
    """

    def __init__(
        self,
        lambda1: float,
        gamma: float,
        c_delta_n: np.ndarray,
    ) -> None:
        """Store penalties and precompute ``C_{delta,n}^{1/2}``.

        Uses the same eigendecomposition strategy as
        :class:`Covridge` (see its docstring for details).

        Args:
            lambda1: Geometry-aware penalty weight ``lambda1``.  Must be
                ``>= 0``.
            gamma: Sparsity weight ``gamma``.  Must be ``>= 0``.
            c_delta_n: Stabilized Gram matrix of shape ``(p, p)``.

        Raises:
            ValueError: If ``lambda1`` or ``gamma`` is negative.
        """
        if lambda1 < 0 or gamma < 0:
            raise ValueError("lambda1 and gamma must be non-negative.")
        self.lambda1 = lambda1
        self.gamma = gamma
        eigvals, eigvecs = np.linalg.eigh(c_delta_n)
        # Symmetry guarantees a real eigenbasis; clamp tiny negative
        # eigenvalues to 0 to keep the square root real-valued.
        eigvals = np.maximum(eigvals, 0.0)
        self._c_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def penalty(self, weights: np.ndarray) -> float:
        r"""Return ``lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1``.

        Args:
            weights: Weight matrix.

        Returns:
            Scalar penalty.
        """
        cw = self._c_sqrt @ weights
        return self.lambda1 * float(np.sum(cw**2)) + self.gamma * float(
            np.sum(np.abs(weights))
        )

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        r"""Return ``2 lambda1 C_{delta,n} W + gamma sign(W)``.

        Args:
            weights: Weight matrix.

        Returns:
            Subgradient array, same shape as ``weights``.
        """
        c_mat = self._c_sqrt @ self._c_sqrt
        l2_grad = 2.0 * self.lambda1 * (c_mat @ weights)
        l1_grad = self.gamma * np.sign(weights)
        return l2_grad + l1_grad
