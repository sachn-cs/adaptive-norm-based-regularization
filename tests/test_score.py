"""Comprehensive tests for evaluation metrics."""

import numpy as np

from regulo.score import Balanced, Mae, Metric, Mse, R2, Rmse


def test_metric_is_abstract():
    # Cannot instantiate the abstract base class.
    import pytest

    with pytest.raises(TypeError):
        Metric()


def test_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(Mse()(y_true, y_pred), 0.25)


def test_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(Mae()(y_true, y_pred), 0.5)


def test_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(Rmse()(y_true, y_pred), 0.5)


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert R2()(y, y) == 1.0


def test_r2_constant_true():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert R2()(y_true, y_pred) == 0.0


def test_r2_constant_perfect():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert R2()(y_true, y_pred) == 1.0


def test_balanced_perfect():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert Balanced()(y_true, y_pred) == 1.0


def test_balanced_all_wrong():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 2, 0])
    assert Balanced()(y_true, y_pred) == 0.0


def test_balanced_single_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert Balanced()(y_true, y_pred) == 1.0


def test_balanced_imbalanced():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0])
    expected = (1.0 + 0.5) / 2.0
    assert np.isclose(Balanced()(y_true, y_pred), expected)


def test_all_metrics_satisfy_metric_protocol():
    metrics = [Mse(), Mae(), Rmse(), R2(), Balanced()]
    for m in metrics:
        assert isinstance(m, Metric)
        assert callable(m)
        assert isinstance(m.name, str) and m.name
