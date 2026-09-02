"""Adam optimizer implemented in pure NumPy.

Implements the Adam algorithm (Kingma & Ba, 2014) with per-parameter
first- and second-moment estimates, bias-corrected step sizes, and an
explicit :meth:`reset` for re-training from scratch.

The optimizer operates on dictionaries of parameter lists (matching the
format returned by :meth:`~regulo.network.FullyConnectedNetwork.get_params`)
so it is agnostic to network architecture.

References
----------
Kingma, D. P. & Ba, J. (2014).  "Adam: A Method for Stochastic
Optimization."  *ICLR 2015*.  arXiv:1412.6980.
"""

from typing import Dict, List, Optional

import numpy as np


class Adam:
    r"""Adam optimizer with adaptive learning rates.

    Maintains exponential moving averages of past gradients (first
    moment, ``m``) and past squared gradients (second moment, ``v``).
    At each step the bias-corrected estimates are used:

    .. math::

        \hat{m}_t &= m_t / (1 - \beta_1^t) \\
        \hat{v}_t &= v_t / (1 - \beta_2^t) \\
        \theta_t   &= \theta_{t-1} - \eta \, \hat{m}_t /
                       (\sqrt{\hat{v}_t} + \epsilon)

    Default hyperparameters (``lr=1e-3``, ``beta1=0.9``, ``beta2=0.999``,
    ``eps=1e-8``) match the original paper.

    Attributes:
        learning_rate: Step size ``eta``.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        epsilon: Small constant added to the denominator for numerical
            stability.
        t: Internal step counter (incremented on each :meth:`step` call).

    Thread safety
    -------------
    The optimizer maintains mutable internal state (``_m``, ``_v``, ``t``).
    It is **not** safe to call :meth:`step` concurrently.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialise Adam hyper-parameters and empty moment buffers.

        Args:
            learning_rate: Step size ``eta`` (default ``1e-3``).
            beta1: Decay rate for the first moment (default ``0.9``).
            beta2: Decay rate for the second moment (default ``0.999``).
            epsilon: Numerical stability term (default ``1e-8``).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self._m: Dict[str, List[Optional[np.ndarray]]] = {}
        self._v: Dict[str, List[Optional[np.ndarray]]] = {}

    def step(
        self,
        params: Dict[str, List[np.ndarray]],
        grads: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        """Perform one Adam update and return the new parameters.

        Lazy moment buffers
        -------------------
        On first contact with each parameter array the corresponding
        ``m`` and ``v`` buffers are lazily created as zero arrays of
        identical shape.  This means no prior ``setup`` step is
        required -- the optimizer is usable the moment it is
        instantiated, regardless of the architecture being trained.

        The step counter ``t`` is incremented *before* the bias
        correction is computed, so that ``t = 1`` on the very first
        call.  The first step therefore uses
        ``m_hat = m / (1 - beta1)`` and ``v_hat = v / (1 - beta2)``,
        which is the correct bias-correction formula in the paper.

        Side effects
        ------------
        Increments ``self.t``, mutates ``self._m`` and ``self._v``.
        The input ``params`` and ``grads`` dictionaries are **not**
        modified -- only the returned dictionary contains the updated
        arrays.

        Complexity
        ----------
        ``O(sum of parameter sizes)``: one element-wise pass over every
        gradient and parameter array.  No allocations beyond the
        updated parameter arrays and the first-time moment buffers.

        Args:
            params: Current parameter arrays, keyed by group name
                (``"weights"``, ``"biases"``).  Values are lists of
                numpy arrays (one per layer or group).
            grads: Gradient arrays with the same structure as *params*.

        Returns:
            A new parameter dictionary with the same structure as
            *params*, containing the post-update arrays.  Callers are
            expected to overwrite their parameters with the returned
            values (see :meth:`~regulo.network.FullyConnectedNetwork.set_params`).
        """
        self.t += 1
        updated: Dict[str, List[np.ndarray]] = {}
        for key in params:
            updated[key] = []
            if key not in self._m:
                self._m[key] = []
                self._v[key] = []
            # Lazily extend the moment lists to match the number of
            # parameter arrays, so callers can add or remove layers
            # between calls without resetting the optimizer.
            while len(self._m[key]) < len(params[key]):
                self._m[key].append(None)
                self._v[key].append(None)
            for i, (p, g) in enumerate(zip(params[key], grads[key])):
                # First contact with this parameter slot: lazily create
                # matching-shape zero moment buffers.
                if self._m[key][i] is None:
                    self._m[key][i] = np.zeros_like(p)
                    self._v[key][i] = np.zeros_like(p)
                m = self._m[key][i]
                v = self._v[key][i]
                # The asserts narrow the type for mypy/strict checkers;
                # the assignment two lines above already guaranteed
                # non-None, so these can never fail at runtime.
                assert m is not None
                assert v is not None
                # Update biased moments (exponential moving averages).
                m = self.beta1 * m + (1.0 - self.beta1) * g
                v = self.beta2 * v + (1.0 - self.beta2) * (g**2)
                self._m[key][i] = m
                self._v[key][i] = v
                # Bias-corrected estimates compensate for the
                # zero-initialization of m and v.
                m_hat = m / (1.0 - self.beta1**self.t)
                v_hat = v / (1.0 - self.beta2**self.t)
                p_new = p - self.learning_rate * m_hat / (
                    np.sqrt(v_hat) + self.epsilon
                )
                updated[key].append(p_new)
        return updated

    def reset(self) -> None:
        """Reset all internal state to the initial condition.

        Clears the step counter and all moment buffers so the optimizer
        can be reused for a fresh training run.  This is useful for
        re-training from scratch without creating a new optimizer
        instance.
        """
        self.t = 0
        self._m = {}
        self._v = {}
