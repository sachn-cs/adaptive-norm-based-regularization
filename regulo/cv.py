"""Cross-validation and hyperparameter grid search utilities.

Provides helpers for building regularizer instances from method names
and hyperparameter dictionaries, and for running exhaustive k-fold
cross-validation over a parameter grid.

Data leakage prevention
-----------------------
Each fold standardizes features independently using
:class:`sklearn.preprocessing.StandardScaler` fit on the training
partition only.  This prevents information leakage from the validation
set into the training process.

Typical workflow
----------------
1. Build a list of candidate hyperparameter dictionaries.
2. Call :func:`grid_search_cv` which internally instantiates fresh
   networks, regularizers, and optimizers for each fold and each grid
   point, returning the best configuration.

Performance note
----------------
``grid_search_cv`` is compute-intensive: it trains ``len(param_grid) *
n_splits`` independent models.  For the paper's grid of 5 values with
Covridge/Sparridge (25 combinations), this means 125 model fits per
CV run.  Consider reducing ``epochs`` or ``n_splits`` during development.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import regulo.losses as losses
import regulo.metrics as metrics
from regulo.network import FullyConnectedNetwork
from regulo.optimizer import Adam
from regulo.regularizers import (
    Covridge,
    ElasticNet,
    Lasso,
    NoRegularizer,
    Regularizer,
    Ridge,
    Sparridge,
)
from regulo.trainer import Trainer

# Expected hyperparameter keys for each regularizer method.
_METHOD_HP_KEYS: Dict[str, List[str]] = {
    "none": [],
    "ridge": ["lambda_"],
    "lasso": ["gamma"],
    "elastic_net": ["alpha", "gamma"],
    "covridge": ["lambda1", "lambda2"],
    "sparridge": ["lambda1", "gamma"],
}


def build_regularizer(
    method: str,
    hp: Dict[str, float],
    x_train: np.ndarray,
    delta: float = 1e-4,
) -> Regularizer:
    r"""Instantiate a regularizer from a method name and hyperparameters.

    For geometry-aware methods (``"covridge"``, ``"sparridge"``) the
    stabilized Gram matrix ``C_{delta,n}`` is computed from *x_train*
    and passed to the constructor.  The Gram matrix is:

    .. math::

        C_{\\delta,n} = \\frac{1}{n} X^T X + \\delta I_p

    where ``n = x_train.shape[0]`` and ``p = x_train.shape[1]``.

    Args:
        method: One of ``"none"``, ``"ridge"``, ``"lasso"``,
            ``"elastic_net"``, ``"covridge"``, ``"sparridge"``.
        hp: Hyperparameter dictionary.  Expected keys depend on *method*:

            * ``"none"``: empty dict.
            * ``"ridge"``: ``{"lambda_": float}``.
            * ``"lasso"``: ``{"gamma": float}``.
            * ``"elastic_net"``: ``{"alpha": float, "gamma": float}``.
            * ``"covridge"``: ``{"lambda1": float, "lambda2": float}``.
            * ``"sparridge"``: ``{"lambda1": float, "gamma": float}``.

        x_train: Training features used to compute ``C_{delta,n}``.
            Shape ``(n, p)``.  Ignored for non-geometry-aware methods.
        delta: Diagonal stabilization constant (default ``1e-4``).
            Larger values make the Gram matrix better-conditioned but
            reduce the geometry-awareness of Covridge/Sparridge.

    Returns:
        A :class:`~regulo.regularizers.Regularizer` instance.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method == "none":
        return NoRegularizer()
    if method == "ridge":
        return Ridge(lambda_=hp["lambda_"])
    if method == "lasso":
        return Lasso(gamma=hp["gamma"])
    if method == "elastic_net":
        return ElasticNet(alpha=hp["alpha"], gamma=hp["gamma"])

    # Geometry-aware regularizers need the empirical Gram matrix.
    n, p = x_train.shape
    c_n = (x_train.T @ x_train) / n
    c_delta_n = c_n + delta * np.eye(p)

    if method == "covridge":
        return Covridge(
            lambda1=hp["lambda1"], lambda2=hp["lambda2"], c_delta_n=c_delta_n
        )
    if method == "sparridge":
        return Sparridge(
            lambda1=hp["lambda1"], gamma=hp["gamma"], c_delta_n=c_delta_n
        )
    raise ValueError(f"Unknown method: {method}")


def grid_search_cv(
    x: np.ndarray,
    y: np.ndarray,
    layer_sizes: List[int],
    method: str,
    param_grid: List[Dict[str, float]],
    loss_fn: losses.MSELoss | losses.CrossEntropyLoss,
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    early_stopping: bool = False,
    patience: int = 10,
    task: str = "regression",
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, float], float]:
    """Run k-fold cross-validation over a hyperparameter grid.

    Each combination in *param_grid* is evaluated on *n_splits* folds.
    A fresh network, regularizer, and optimizer are created per fold to
    avoid state leakage, and each fold trains for ``epochs`` epochs
    (unless ``early_stopping`` truncates sooner).

    Scoring metric
    --------------
    The scoring metric is selected via *task*:

    * ``"regression"`` -- **negative MSE** (higher is better; matches
      sklearn's "greater is better" convention).
    * ``"classification"`` -- **balanced accuracy** (higher is better).

    Within a single grid point the metric is averaged across folds;
    across grid points the best mean metric wins.

    Cost
    ----
    Training budget is ``len(param_grid) * n_splits`` independent
    model fits, each running for up to ``epochs`` epochs.  For the
    paper's 5-value ``SIM_GRID`` with Covridge/Sparridge (25
    combinations), ``grid_search_cv`` performs 125 model fits per CV
    run.  Consider smaller grids or ``early_stopping=True`` during
    development; the full grid with ``epochs=500`` is reserved for the
    final experimental runs.

    Args:
        x: Feature matrix of shape ``(n, p)``.
        y: Target array of shape ``(n, 1)`` (regression) or ``(n,)``
            (classification).
        layer_sizes: Network architecture widths including input and
            output.
        method: Regularization method name (see :func:`build_regularizer`).
        param_grid: List of hyperparameter dictionaries to evaluate.
                An empty list is accepted but the function will fail on
                ``max`` of an empty sequence -- callers should pass at
                least one configuration.
        loss_fn: Loss function instance.
        n_splits: Number of CV folds (default ``5``).
        batch_size: Mini-batch size (default ``32``).
        epochs: Training epochs per fold (default ``500``).
        learning_rate: Adam step size (default ``1e-3``).
        early_stopping: Enable early stopping within each fold.
        patience: Early-stopping patience (default ``10``).
        task: ``"regression"`` or ``"classification"``.
        random_state: Seed for the ``KFold`` splitter.  Pass an int for
            reproducibility; ``None`` defers to NumPy's default seeding.

    Returns:
        ``(best_params, best_score)``:

        * ``best_params`` -- the dictionary (a copy of the winning
          grid entry) that achieved the highest mean fold score.
          Empty when *method* is ``"none"``.
        * ``best_score`` -- the corresponding mean score: negative MSE
          for ``task="regression"``, balanced accuracy for
          ``task="classification"``.

    Raises:
        ValueError: If *task* is not one of the recognized strings.
        KeyError: Propagated from the chosen regularizer when a
            hyperparameter key required by *method* is missing from a
            grid entry.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -float("inf")
    best_params: Optional[Dict[str, float]] = None

    for params in param_grid:
        scores: List[float] = []
        for train_idx, val_idx in kf.split(x):
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Standardize within each fold to prevent leakage.
            scaler = StandardScaler()
            x_train_fold = scaler.fit_transform(x_train_fold)
            x_val_fold = scaler.transform(x_val_fold)

            regularizer = build_regularizer(method, params, x_train_fold)
            net = FullyConnectedNetwork(layer_sizes)
            opt = Adam(learning_rate=learning_rate)
            trainer = Trainer(
                net,
                loss_fn,
                regularizer,
                opt,
                batch_size=batch_size,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )
            trainer.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
            preds = trainer.predict(x_val_fold)

            if task == "regression":
                score = -metrics.mean_squared_error(y_val_fold, preds)
            else:
                class_preds = np.argmax(preds, axis=1)
                score = metrics.balanced_accuracy_score(y_val_fold, class_preds)
            scores.append(score)

        avg_score = float(np.mean(scores))
        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

    assert best_params is not None
    return best_params, best_score
