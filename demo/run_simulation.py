"""Replicate simulation experiments (Tables 1--3) on a reduced scale.

Full paper uses 100 Monte Carlo replications.  By default this script runs
5 replications for speed; pass ``--full`` for 100.
"""

import argparse
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler

import regulo.losses as losses
import regulo.metrics as metrics
from regulo.cv import build_regularizer, grid_search_cv
from regulo.data import make_dgp1, make_dgp2
from regulo.network import FullyConnectedNetwork
from regulo.optimizer import Adam
from regulo.trainer import Trainer

METHODS = ["none", "ridge", "lasso", "elastic_net", "covridge", "sparridge"]

SIM_GRID = [0.001, 0.01, 0.1, 0.5, 0.9]


def _build_param_grid(method: str) -> List[Dict[str, float]]:
    """Return the hyperparameter search grid for *method*.

    Args:
        method: Regularisation method name.

    Returns:
        List of hyperparameter dictionaries.
    """
    if method == "none":
        return [{}]
    if method == "ridge":
        return [{"lambda_": v} for v in SIM_GRID]
    if method == "lasso":
        return [{"gamma": v} for v in SIM_GRID]
    if method == "elastic_net":
        return [{"alpha": a, "gamma": g} for a in [0.5] for g in SIM_GRID]
    if method == "covridge":
        return [
            {"lambda1": a, "lambda2": b} for a in SIM_GRID for b in SIM_GRID
        ]
    if method == "sparridge":
        return [{"lambda1": a, "gamma": g} for a in SIM_GRID for g in SIM_GRID]
    return []


def _evaluate_method(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    layer_sizes: List[int],
    epochs: int = 500,
) -> Dict[str, float]:
    """Cross-validate, retrain, and evaluate a single method.

    Args:
        method: Regularisation method name.
        x_train: Training features.
        y_train: Training targets.
        x_test: Held-out test features.
        y_test: Held-out test targets.
        layer_sizes: Network architecture widths.
        epochs: Training epochs for both CV and final fit.

    Returns:
        Dictionary with ``"mse"`` and ``"mae"`` on the test set.
    """
    param_grid = _build_param_grid(method)
    if len(param_grid) == 1 and param_grid[0] == {}:
        best_params: Dict[str, float] = {}
    else:
        best_params, _ = grid_search_cv(
            x_train,
            y_train,
            layer_sizes,
            method,
            param_grid,
            loss_fn=losses.MSELoss(),
            n_splits=5,
            epochs=epochs,
            learning_rate=1e-3,
        )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    reg = build_regularizer(method, best_params, x_train_s)
    net = FullyConnectedNetwork(layer_sizes)
    opt = Adam(learning_rate=1e-3)
    trainer = Trainer(
        net, losses.MSELoss(), reg, opt, batch_size=32, epochs=epochs
    )
    trainer.fit(x_train_s, y_train)
    preds = trainer.predict(x_test_s)
    return {
        "mse": metrics.mean_squared_error(y_test, preds),
        "mae": metrics.mean_absolute_error(y_test, preds),
    }


def _run_dgp(
    dgp_fn,
    name: str,
    layer_sizes: List[int],
    n_reps: int,
    rho_values: List[float],
    sigma_values: List[float],
) -> None:
    """Run all methods on a DGP across multiple replications.

    Args:
        dgp_fn: Callable ``(rho, sigma_noise, random_state) -> (X, y)``.
        name: Human-readable label for the DGP (used in output).
        layer_sizes: Network architecture widths.
        n_reps: Number of Monte Carlo replications.
        rho_values: Correlation values to sweep.
        sigma_values: Noise levels to sweep.
    """
    for rho in rho_values:
        for sigma in sigma_values:
            print(f"\n{name} -- rho={rho}, sigma={sigma}")
            results: Dict[str, List[float]] = {m: [] for m in METHODS}
            for rep in range(n_reps):
                x, y = dgp_fn(rho=rho, sigma_noise=sigma, random_state=rep)
                n = x.shape[0]
                split = int(0.75 * n)
                x_train, x_test = x[:split], x[split:]
                y_train, y_test = y[:split], y[split:]
                for method in METHODS:
                    res = _evaluate_method(
                        method, x_train, y_train, x_test, y_test, layer_sizes
                    )
                    results[method].append(res["mse"])
            print(f"{'Method':<15} {'MSE mean':<12} {'MSE std':<12}")
            for method in METHODS:
                arr = np.array(results[method])
                print(
                    f"{method:<15} {np.mean(arr):<12.4f} {np.std(arr):<12.4f}"
                )


def main() -> None:
    """Entry point: parse CLI args and run DGP experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Run 100 replications (slow)."
    )
    args = parser.parse_args()
    n_reps = 100 if args.full else 5

    print("=== Simulation Reproduction ===")
    print(f"Replications: {n_reps}")

    # DGP1
    _run_dgp(
        lambda rho, sigma_noise, random_state: make_dgp1(
            rho=rho, sigma_noise=sigma_noise, random_state=random_state
        ),
        "DGP1 (linear)",
        [20, 64, 32, 1],
        n_reps,
        [0.25, 0.75],
        [0.10, 2.00],
    )
    _run_dgp(
        lambda rho, sigma_noise, random_state: make_dgp1(
            rho=rho,
            sigma_noise=sigma_noise,
            nonlinear=True,
            random_state=random_state,
        ),
        "DGP1 (nonlinear)",
        [20, 64, 32, 1],
        n_reps,
        [0.25, 0.75],
        [0.10, 2.00],
    )

    # DGP2 (reduced to 1 rep for speed unless --full)
    if args.full:
        _run_dgp(
            lambda rho, sigma_noise, random_state: make_dgp2(
                rho=rho, sigma_noise=sigma_noise, random_state=random_state
            ),
            "DGP2 (linear)",
            [200, 64, 32, 1],
            n_reps,
            [0.25, 0.75],
            [0.10, 2.00],
        )
    else:
        print("\n(Skipping DGP2/DGP3 in quick mode; use --full to run.)")


if __name__ == "__main__":
    main()
