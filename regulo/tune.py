"""Cross-validation and hyperparameter grid search utilities.

Provides primitives for k-fold splitting, feature standardization,
penalty construction from method-name dispatch, and exhaustive
grid search over hyperparameter dictionaries.
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import Penalty, REGISTRY

__all__ = ["kfold", "Scaler", "resolve", "search"]


def kfold(
    n: int, n_splits: int, seed: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, val_idx)`` tuples for *n_splits* folds.

    Each fold uses a contiguous slice of a shuffled permutation;
    shuffling is seeded by *seed* for reproducibility.  Standard
    K-fold behaviour with shuffle.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > n:
        raise ValueError("n_splits cannot exceed n.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1
    start = 0
    for size in fold_sizes:
        val_idx = perm[start : start + size]
        train_idx = np.concatenate([perm[:start], perm[start + size :]])
        yield train_idx, val_idx
        start += size


class Scaler:
    """Per-column z-score scaler fit on one array, applied to others.

    Computes and stores the mean and standard deviation of each
    column on :meth:`fit`, then applies the transformation via
    :meth:`transform`.  :meth:`fit_transform` combines the two for
    convenience.
    """

    name = "scaler"

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "Scaler":
        """Compute per-column mean and standard deviation."""
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        # Guard against constant columns (std == 0) so transform does
        # not produce NaN.
        if self.std is not None:
            self.std = np.where(self.std == 0.0, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit on *x* and return the transformed array."""
        self.fit(x)
        return self.transform(x)

    def __repr__(self) -> str:
        if self.mean is None:
            return "Scaler(unfit)"
        return f"Scaler(mean={self.mean.tolist()!r}, std={self.std.tolist()!r})"


def resolve(
    name: str,
    hp: dict,
    x_train: np.ndarray,
    delta: float = 1e-4,
) -> Penalty:
    """Construct a :class:`Penalty` from a method name and hyperparameters.

    For geometry-aware methods (``"covridge"``, ``"sparridge"``)
    the stabilized Gram matrix ``C_{delta,n}`` is computed from
    *x_train* and passed to the constructor.  The Gram matrix is
    also computed unconditionally for non-geometry methods; the
    small overhead is the price of dispatch simplicity.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown method: {name!r}. Known: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    unknown = set(hp) - set(cls.hp)
    if unknown:
        raise ValueError(
            f"Method {name!r} accepts keys {cls.hp}, "
            f"but got unexpected {sorted(unknown)}."
        )
    n, p = x_train.shape
    gram = (x_train.T @ x_train) / n + delta * np.eye(p)
    hp = dict(hp)
    if "c_delta_n" in cls.hp:
        hp.setdefault("c_delta_n", gram)
    return cls(**hp)


def search(
    x: np.ndarray,
    y: np.ndarray,
    layer_sizes: List[int],
    method: str,
    param_grid: List[dict],
    loss_fn: Loss,
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    early_stopping: bool = False,
    patience: int = 10,
    seed: Optional[int] = None,
) -> Tuple[Optional[dict], float]:
    """Run k-fold cross-validation over a hyperparameter grid.

    Each combination in *param_grid* is evaluated on *n_splits*
    folds.  A fresh network, penalty, and optimizer are created
    per fold to avoid state leakage.

    The scoring metric is derived from ``loss_fn``: negative MSE for
    :class:`regulo.loss.Square`, balanced accuracy for
    :class:`regulo.loss.Softmax`.

    Returns:
        ``(best_params, best_score)`` -- the hyperparameters that
        achieved the highest mean fold score and that score.
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one entry.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    task = "classification" if loss_fn.name == "softmax" else "regression"
    best_score = -float("inf")
    best_params: Optional[dict] = None
    from regulo.score import Balanced, Mse

    metric = Balanced() if task == "classification" else Mse()

    for params in param_grid:
        scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold(x.shape[0], n_splits, seed=seed)
        ):
            x_train_fold = x[train_idx]
            x_val_fold = x[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            scaler = Scaler().fit(x_train_fold)
            x_train_fold = scaler.transform(x_train_fold)
            x_val_fold = scaler.transform(x_val_fold)

            penalty = resolve(method, params, x_train_fold)
            mlp = MLP(layer_sizes, seed=seed)
            adam = Adam(learning_rate=learning_rate)
            runner = Runner(
                mlp,
                loss_fn,
                penalty,
                adam,
                batch_size=batch_size,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )
            runner.fit(
                x_train_fold,
                y_train_fold,
                x_val_fold,
                y_val_fold,
                seed=seed,
            )
            preds = runner.predict(x_val_fold)
            if task == "classification":
                class_preds = runner.predict_class(x_val_fold)
                score = metric(y_val_fold, class_preds)
            else:
                score = -metric(y_val_fold, preds)
            scores.append(score)

        avg = float(np.mean(scores))
        if avg > best_score:
            best_score = avg
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("No grid point evaluated.")
    return best_params, best_score
