"""Replicate simulation experiments (Tables 1-3) on a reduced scale.

The paper uses 100 Monte Carlo replications.  By default this script
runs 5 replications for speed; pass ``--full`` for 100.

Run from the repository root after installing regulo::

    pip install -e .
    python demo/run_simulation.py --seed 0
"""

import argparse
from typing import Dict, List

import numpy as np

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square
from regulo.net import MLP
from regulo.penalty import ElasticNet, Lasso, Ridge, Sparridge, Void
from regulo.score import Mse
from regulo.tune import Scaler, resolve, search

METHODS = ["none", "ridge", "lasso", "elastic_net", "covridge", "sparridge"]

# A reduced grid for the demo; the full paper grid is {0.001, 0.01,
# 0.1, 0.5, 0.9} (25 grid points per geometry-aware method).
SIM_GRID = [0.01, 0.1, 0.5]


def build_param_grid(method: str) -> List[Dict[str, float]]:
    """Return the hyperparameter search grid for *method*."""
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


def evaluate(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    layer_sizes: List[int],
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    """Cross-validate, retrain, and evaluate a single method."""
    param_grid = build_param_grid(method)
    if len(param_grid) == 1 and param_grid[0] == {}:
        best_params: Dict[str, float] = {}
    else:
        best_params, _ = search(
            x_train,
            y_train,
            layer_sizes,
            method,
            param_grid,
            loss_fn=Square(),
            n_splits=3,
            epochs=max(50, epochs // 2),
            learning_rate=1e-3,
            seed=seed,
        )

    scaler = Scaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    penalty = resolve(method, best_params, x_train_s)
    mlp = MLP(layer_sizes, seed=seed)
    adam = Adam(learning_rate=1e-3)
    runner = Runner(
        mlp,
        Square(),
        penalty,
        adam,
        batch_size=32,
        epochs=epochs,
    )
    runner.fit(x_train_s, y_train, seed=seed)
    preds = runner.predict(x_test_s)
    return {
        "mse": Mse()(y_test, preds),
    }


def shuffle_split(
    x: np.ndarray, y: np.ndarray, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random 75/25 (or other) train/test split using numpy."""
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int((1.0 - test_frac) * n)
    return x[perm[:cut]], x[perm[cut:]], y[perm[:cut]], y[perm[cut:]]


def run_dgp(
    dgp_factory,
    name: str,
    layer_sizes: List[int],
    n_reps: int,
    rho_values: List[float],
    sigma_values: List[float],
    seed: int,
) -> None:
    """Run all methods on a DGP across multiple replications."""
    for rho in rho_values:
        for sigma in sigma_values:
            print(f"\n{name} -- rho={rho}, sigma={sigma}")
            results: Dict[str, List[float]] = {m: [] for m in METHODS}
            for rep in range(n_reps):
                x, y = dgp_factory(rho, sigma, rep)
                x_train, x_test, y_train, y_test = shuffle_split(
                    x, y, test_frac=0.25, seed=seed + rep
                )
                for method in METHODS:
                    res = evaluate(
                        method,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        layer_sizes,
                        epochs=100,
                        seed=seed + rep,
                    )
                    results[method].append(res["mse"])
            print(f"{'Method':<15} {'MSE mean':<12} {'MSE std':<12}")
            for method in METHODS:
                arr = np.array(results[method])
                print(
                    f"{method:<15} {np.mean(arr):<12.4f} {np.std(arr):<12.4f}"
                )


def make_dgp(rho: float, sigma_noise: float, rep: int) -> tuple[np.ndarray, np.ndarray]:
    """DGP1 wrapper for the linear signal."""
    from regulo.data import synth

    return synth(
        n=200, p=20, k=10,
        rho=rho, sigma_noise=sigma_noise,
        nonlinear=False, seed=rep,
    )


def make_dgp_nonlinear(rho: float, sigma_noise: float, rep: int) -> tuple[np.ndarray, np.ndarray]:
    """DGP1 wrapper for the nonlinear (sinusoidal) signal."""
    from regulo.data import synth

    return synth(
        n=200, p=20, k=10,
        rho=rho, sigma_noise=sigma_noise,
        nonlinear=True, seed=rep,
    )


def main() -> None:
    """Entry point: parse CLI args and run DGP experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Run 100 replications (slow)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)."
    )
    args = parser.parse_args()
    n_reps = 100 if args.full else 5
    seed = args.seed

    print("=== Simulation Reproduction ===")
    print(f"Replications: {n_reps}")
    print(f"Seed: {seed}")

    run_dgp(
        make_dgp,
        "DGP1 (linear)",
        [20, 64, 32, 1],
        n_reps,
        [0.25, 0.75],
        [0.10, 2.00],
        seed,
    )
    run_dgp(
        make_dgp_nonlinear,
        "DGP1 (nonlinear)",
        [20, 64, 32, 1],
        n_reps,
        [0.25, 0.75],
        [0.10, 2.00],
        seed,
    )


if __name__ == "__main__":
    main()
