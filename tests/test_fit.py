"""Integration tests: end-to-end training with all penalties."""

import numpy as np
import pytest

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square, Softmax
from regulo.net import MLP
from regulo.penalty import (
    Covridge,
    ElasticNet,
    Lasso,
    Ridge,
    Sparridge,
    Void,
)
from regulo.score import Balanced, Mse


def test_fit_regression_decreases_loss():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 5))
    y = (
        x @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]).reshape(-1, 1)
        + rng.standard_normal((64, 1)) * 0.1
    )
    runner = Runner(
        MLP([5, 8, 1]),
        Square(),
        Ridge(lambda_=1e-4),
        Adam(learning_rate=1e-2),
        batch_size=16,
        epochs=100,
    )
    runner.fit(x, y, seed=0)
    assert runner.history["train"][-1] < runner.history["train"][0]
    assert runner.predict(x).shape == y.shape


def test_classification_forward_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 4))
    y = rng.integers(0, 3, size=(32,))
    runner = Runner(
        MLP([4, 8, 3]),
        Softmax(),
        Ridge(lambda_=1e-4),
        Adam(learning_rate=1e-2),
        batch_size=16,
        epochs=50,
    )
    runner.fit(x, y, seed=0)
    logits = runner.predict(x)
    assert logits.shape == (32, 3)
    preds = runner.predict_class(x)
    assert preds.shape == (32,)
    assert 0.0 <= Balanced()(y, preds) <= 1.0


def test_train_with_each_penalty():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 4))
    y = rng.standard_normal((40, 1))
    c = np.eye(4)
    penalties = [
        Void(),
        Ridge(lambda_=1e-4),
        Lasso(gamma=1e-4),
        ElasticNet(alpha=0.5, gamma=1e-4),
        Covridge(lambda1=1e-4, lambda2=1e-4, c_delta_n=c),
        Sparridge(lambda1=1e-4, gamma=1e-4, c_delta_n=c),
    ]
    for penalty in penalties:
        runner = Runner(
            MLP([4, 4, 1]),
            Square(),
            penalty,
            Adam(learning_rate=1e-2),
            batch_size=16,
            epochs=20,
        )
        runner.fit(x, y, seed=0)
        assert len(runner.history["train"]) == 20


def test_early_stopping():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 4))
    y = rng.standard_normal((64, 1))
    runner = Runner(
        MLP([4, 4, 1]),
        Square(),
        Ridge(lambda_=1e-4),
        Adam(learning_rate=1e-2),
        batch_size=16,
        epochs=500,
        early_stopping=True,
        patience=5,
    )
    runner.fit(x[:40], y[:40], x[40:], y[40:], seed=0)
    assert len(runner.history["train"]) < 500


def test_feature_standardization_consistency():
    from regulo.tune import Scaler

    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 4))
    y = rng.standard_normal((100, 1))
    scaler = Scaler().fit(x[:80])
    x_train = scaler.transform(x[:80])
    x_test = scaler.transform(x[80:])
    runner = Runner(
        MLP([4, 4, 1]),
        Square(),
        Ridge(lambda_=1e-4),
        Adam(),
        batch_size=16,
        epochs=10,
    )
    runner.fit(x_train, y[:80], seed=0)
    assert runner.predict(x_test).shape == (20, 1)


def test_history_keys():
    runner = Runner(
        MLP([3, 2, 1]),
        Square(),
        Ridge(lambda_=1e-4),
        Adam(),
        epochs=5,
    )
    x = np.random.randn(32, 3)
    y = np.random.randn(32, 1)
    runner.fit(x, y, seed=0)
    assert "train" in runner.history
    assert len(runner.history["train"]) == 5
    assert "val" in runner.history
    assert len(runner.history["val"]) == 0


def test_history_with_validation():
    runner = Runner(
        MLP([3, 2, 1]),
        Square(),
        Ridge(lambda_=1e-4),
        Adam(),
        epochs=5,
    )
    x = np.random.randn(32, 3)
    y = np.random.randn(32, 1)
    runner.fit(x[:20], y[:20], x[20:], y[20:], seed=0)
    assert len(runner.history["val"]) == 5


def test_task_derived_from_loss():
    assert Runner(MLP([3, 1]), Square(), Void(), Adam()).task == "regression"
    assert Runner(MLP([3, 2]), Softmax(), Void(), Adam()).task == "classification"


def test_predict_proba_for_regression_raises():
    runner = Runner(MLP([3, 1]), Square(), Void(), Adam())
    with pytest.raises(NotImplementedError):
        runner.predict_proba(np.zeros((1, 3)))


def test_predict_proba_sums_to_one():
    runner = Runner(MLP([3, 4]), Softmax(), Void(), Adam(), epochs=1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3))
    y = rng.integers(0, 4, size=(8,))
    runner.fit(x, y, seed=0)
    proba = runner.predict_proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_reset_reinitializes_network():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lambda_=0.01),
        Adam(),
        epochs=2,
    )
    runner.fit(x, y, seed=0)
    runner.reset(seed=0)
    assert runner.history["train"] == []
    fresh = MLP([3, 4, 1], seed=0)
    for w_runner, w_fresh in zip(runner.mlp.weights, fresh.weights):
        np.testing.assert_array_equal(w_runner, w_fresh)
    assert runner.adam.clock == 0


def test_runner_reproducible_with_seed():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = rng.standard_normal((40, 1))
    r1 = Runner(
        MLP([3, 8, 1], seed=42), Square(), Ridge(0.001), Adam(), epochs=5, batch_size=8
    )
    r2 = Runner(
        MLP([3, 8, 1], seed=42), Square(), Ridge(0.001), Adam(), epochs=5, batch_size=8
    )
    r1.fit(x, y, seed=123)
    r2.fit(x, y, seed=123)
    np.testing.assert_allclose(r1.history["train"], r2.history["train"])


def test_non_finite_weights_raise():
    # Inject a NaN into the network weights directly; the guard
    # should detect it on the next step.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 3))
    y = rng.standard_normal((10, 1))
    runner = Runner(
        MLP([3, 4, 1]),
        Square(),
        Void(),
        Adam(),
        batch_size=4,
        epochs=2,
    )
    # Corrupt weights so the next step produces non-finite updates.
    runner.mlp.weights[0][0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        runner.fit(x, y, seed=0)


def test_early_stopping_requires_validation_data():
    runner = Runner(
        MLP([3, 2, 1]),
        Square(),
        Ridge(lambda_=0.001),
        Adam(),
        epochs=2,
        early_stopping=True,
    )
    with pytest.raises(ValueError):
        runner.fit(np.zeros((10, 3)), np.zeros((10, 1)))
