"""Comprehensive tests for evaluation metrics."""

import numpy as np
import pytest

from regulo.score import Balanced, Mae, Metric, Mse, R2, Rmse


def test_metric_abstract():
    with pytest.raises(TypeError):
        Metric()


def test_mse():
    truth = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(Mse()(truth, pred), 0.25)


def test_mae():
    truth = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(Mae()(truth, pred), 0.5)


def test_rmse():
    truth = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(Rmse()(truth, pred), 0.5)


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert R2()(y, y) == 1.0


def test_r2_constant():
    truth = np.array([2.0, 2.0, 2.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert R2()(truth, pred) == 0.0


def test_r2_constant_perfect():
    truth = np.array([2.0, 2.0, 2.0])
    pred = np.array([2.0, 2.0, 2.0])
    assert R2()(truth, pred) == 1.0


def test_balanced_perfect():
    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    assert Balanced()(truth, pred) == 1.0


def test_balanced_all_wrong():
    truth = np.array([0, 1, 2])
    pred = np.array([1, 2, 0])
    assert Balanced()(truth, pred) == 0.0


def test_balanced_single_class():
    truth = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])
    assert Balanced()(truth, pred) == 1.0


def test_balanced_imbalanced():
    truth = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0, 1, 0])
    expected = (1.0 + 0.5) / 2.0
    assert np.isclose(Balanced()(truth, pred), expected)


def test_protocol():
    metrics = [Mse(), Mae(), Rmse(), R2(), Balanced()]
    for m in metrics:
        assert isinstance(m, Metric)
        assert callable(m)
        assert isinstance(m.name, str) and m.name


def test_repr():
    assert repr(Mse()) == "Mse()"
    assert repr(Mae()) == "Mae()"
