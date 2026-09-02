"""Tests for k-fold splitting, standardization, and grid search."""

import numpy as np
import pytest

from regulo.tune import Scaler, kfold, resolve, search


def test_kfold_yields_correct_number_of_folds():
    folds = list(kfold(100, 5, seed=0))
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert len(train_idx) + len(val_idx) == 100


def test_kfold_indices_partition_samples():
    folds = list(kfold(50, 5, seed=0))
    val_union = np.concatenate([v for _, v in folds])
    assert sorted(val_union.tolist()) == list(range(50))


def test_kfold_reproducible_with_seed():
    a = [v.tolist() for _, v in kfold(50, 5, seed=42)]
    b = [v.tolist() for _, v in kfold(50, 5, seed=42)]
    assert a == b


def test_kfold_rejects_invalid_n_splits():
    with pytest.raises(ValueError):
        list(kfold(100, 1))
    with pytest.raises(ValueError):
        list(kfold(2, 5))


def test_scaler_fit_transform_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 4)) * 3.0 + 5.0
    s = Scaler().fit(x)
    out = s.transform(x)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-10)


def test_scaler_transform_matches_fit_transform():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 3))
    s = Scaler().fit(x[:20])
    np.testing.assert_allclose(
        s.transform(x[:20]), s.fit_transform(x[:20])
    )


def test_scaler_constant_column_does_not_div_zero():
    x = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    out = Scaler().fit_transform(x)
    # Constant column stays zero-mean (after centering), not NaN.
    assert np.all(np.isfinite(out))


def test_scaler_transform_before_fit_raises():
    s = Scaler()
    with pytest.raises(RuntimeError):
        s.transform(np.zeros((3, 2)))


def test_resolve_constructs_ridge():
    x = np.random.randn(50, 4)
    p = resolve("ridge", {"lambda_": 0.01}, x)
    assert p.name == "ridge"
    assert p.lambda_ == 0.01


def test_resolve_constructs_covridge_with_gram():
    x = np.random.randn(50, 4)
    p = resolve("covridge", {"lambda1": 0.1, "lambda2": 0.01}, x)
    assert p.lambda1 == 0.1
    assert p.lambda2 == 0.01
    assert p.csqrt.shape == (4, 4)


def test_resolve_rejects_unknown_method():
    with pytest.raises(ValueError):
        resolve("nonsense", {}, np.zeros((10, 2)))


def test_resolve_rejects_unexpected_hp_keys():
    x = np.zeros((10, 3))
    with pytest.raises(ValueError):
        resolve("ridge", {"lambda_": 0.1, "garbage": 1.0}, x)


def test_search_returns_best_params_and_score():
    from regulo.loss import Square

    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    y = (
        x @ np.array([1.0, -1.0, 0.5, 0.0]).reshape(-1, 1)
        + rng.standard_normal((60, 1)) * 0.1
    )
    param_grid = [{"lambda_": 1e-3}, {"lambda_": 1e-2}]
    best, score = search(
        x,
        y,
        layer_sizes=[4, 4, 1],
        method="ridge",
        param_grid=param_grid,
        loss_fn=Square(),
        n_splits=3,
        epochs=10,
    )
    assert "lambda_" in best
    assert score <= 0  # negative MSE


def test_search_rejects_empty_param_grid():
    from regulo.loss import Square

    with pytest.raises(ValueError):
        search(
            np.zeros((10, 2)),
            np.zeros((10, 1)),
            layer_sizes=[2, 1],
            method="ridge",
            param_grid=[],
            loss_fn=Square(),
        )


def test_search_reproducible_with_seed():
    from regulo.loss import Square

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = rng.standard_normal((40, 1))
    grid = [{"lambda_": 1e-2}]
    _, score_a = search(
        x, y, [3, 4, 1], "ridge", grid, Square(), n_splits=3, epochs=5, seed=7
    )
    _, score_b = search(
        x, y, [3, 4, 1], "ridge", grid, Square(), n_splits=3, epochs=5, seed=7
    )
    assert score_a == score_b
