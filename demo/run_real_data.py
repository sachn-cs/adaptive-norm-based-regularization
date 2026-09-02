"""Replicate real-data experiments: UCI Energy and GSE9476 Leukemia.

Runs all six regularisation methods on the UCI Energy Efficiency dataset
(regression, Table 4) and the GSE9476 Leukemia microarray dataset
(classification, Table 5).  The leukemia experiment uses repeated
10x5-fold cross-validation as described in the paper.
"""

from typing import Dict, List

import numpy as np

import regulo.losses as losses
import regulo.metrics as metrics
from regulo.cv import build_regularizer, grid_search_cv
from regulo.data import load_energy_data, load_leukemia_data
from regulo.network import FullyConnectedNetwork
from regulo.optimizer import Adam
from regulo.trainer import Trainer

METHODS = ["none", "ridge", "lasso", "elastic_net", "covridge", "sparridge"]
ENERGY_GRID = [0.001, 0.01, 0.1, 0.5, 0.9]
LEUKEMIA_BOUNDS = [0.0001, 1.0]
LEUKEMIA_GRID = [
    float(x) for x in np.linspace(LEUKEMIA_BOUNDS[0], LEUKEMIA_BOUNDS[1], 5)
]


def _build_param_grid(method: str, grid: List[float]) -> List[Dict[str, float]]:
    """Return the hyperparameter search grid for *method* given *grid* values.

    Args:
        method: Regularisation method name.
        grid: List of candidate hyperparameter values.

    Returns:
        List of hyperparameter dictionaries.
    """
    if method == "none":
        return [{}]
    if method == "ridge":
        return [{"lambda_": v} for v in grid]
    if method == "lasso":
        return [{"gamma": v} for v in grid]
    if method == "elastic_net":
        return [{"alpha": 0.5, "gamma": g} for g in grid]
    if method == "covridge":
        return [{"lambda1": a, "lambda2": b} for a in grid for b in grid]
    if method == "sparridge":
        return [{"lambda1": a, "gamma": g} for a in grid for g in grid]
    return []


def run_energy() -> None:
    """Run the UCI Energy Efficiency regression benchmark.

    Performs 5-fold cross-validation for each method, retrains on the
    full training set with the best hyperparameters, and prints MSE,
    MAE, R-squared, and RMSE on the held-out test set.
    """
    print("\n=== UCI Energy Efficiency (Cooling Load) ===")
    x_train, x_test, y_train, y_test, scaler = load_energy_data(
        test_size=0.25, random_state=42
    )
    layer_sizes = [x_train.shape[1], 64, 32, 1]
    print(f"{'Method':<20} {'MSE':<10} {'MAE':<10} {'R2':<10} {'RMSE':<10}")
    for method in METHODS:
        param_grid = _build_param_grid(method, ENERGY_GRID)
        best_params, _ = grid_search_cv(
            x_train,
            y_train,
            layer_sizes,
            method,
            param_grid,
            loss_fn=losses.MSELoss(),
            n_splits=5,
            epochs=500,
            batch_size=32,
            learning_rate=1e-3,
        )
        reg = build_regularizer(method, best_params, x_train)
        net = FullyConnectedNetwork(layer_sizes)
        opt = Adam(learning_rate=1e-3)
        trainer = Trainer(
            net, losses.MSELoss(), reg, opt, batch_size=32, epochs=500
        )
        trainer.fit(x_train, y_train)
        preds = trainer.predict(x_test)
        mse = metrics.mean_squared_error(y_test, preds)
        mae = metrics.mean_absolute_error(y_test, preds)
        r2 = metrics.r2_score(y_test, preds)
        rmse = metrics.root_mean_squared_error(y_test, preds)
        print(
            f"{method:<20} {mse:<10.3f} {mae:<10.3f} {r2:<10.3f}"
            f" {rmse:<10.3f}"
        )


def run_leukemia() -> None:
    """Run the GSE9476 Leukemia classification benchmark.

    Uses repeated 10x5-fold cross-validation with balanced accuracy as
    the evaluation metric.  Prints mean and standard deviation across
    repeats for each method.
    """
    print("\n=== GSE9476 Leukemia Classification ===")
    try:
        x_train, x_test, y_train, y_test, scaler = load_leukemia_data(
            n_features=2000, test_size=0.2, random_state=42
        )
    except RuntimeError as exc:
        print(f"Skipping leukemia experiment: {exc}")
        return
    layer_sizes = [x_train.shape[1], 8, 4, len(np.unique(y_train))]
    print(f"{'Method':<20} {'Balanced Accuracy':<20}")
    scores: Dict[str, List[float]] = {m: [] for m in METHODS}
    # Repeated 10x5-fold CV as in paper.
    n_repeats = 10
    for rep in range(n_repeats):
        for method in METHODS:
            param_grid = _build_param_grid(method, LEUKEMIA_GRID)
            best_params, _ = grid_search_cv(
                x_train,
                y_train,
                layer_sizes,
                method,
                param_grid,
                loss_fn=losses.CrossEntropyLoss(),
                n_splits=5,
                epochs=500,
                batch_size=16,
                learning_rate=1e-3,
                early_stopping=True,
                patience=10,
                task="classification",
                random_state=rep,
            )
            reg = build_regularizer(method, best_params, x_train)
            net = FullyConnectedNetwork(layer_sizes)
            opt = Adam(learning_rate=1e-3)
            trainer = Trainer(
                net,
                losses.CrossEntropyLoss(),
                reg,
                opt,
                batch_size=16,
                epochs=500,
                early_stopping=True,
                patience=10,
            )
            trainer.fit(x_train, y_train, x_test, y_test)
            preds = np.argmax(trainer.predict(x_test), axis=1)
            acc = metrics.balanced_accuracy_score(y_test, preds)
            scores[method].append(acc)
    for method in METHODS:
        arr = np.array(scores[method])
        print(f"{method:<20} {np.mean(arr):<10.4f} +/- {np.std(arr):<10.4f}")


def main() -> None:
    """Entry point: run both Energy and Leukemia experiments."""
    run_energy()
    run_leukemia()


if __name__ == "__main__":
    main()
