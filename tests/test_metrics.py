"""Comprehensive tests for evaluation metrics."""

import numpy as np

from regulo.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def test_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(mean_squared_error(y_true, y_pred), 0.25)


def test_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(mean_absolute_error(y_true, y_pred), 0.5)


def test_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert np.isclose(root_mean_squared_error(y_true, y_pred), 0.5)


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == 1.0


def test_r2_constant_true():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # When y_true is constant, ss_tot=0.
    assert r2_score(y_true, y_pred) == 0.0


def test_r2_constant_perfect():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert r2_score(y_true, y_pred) == 1.0


def test_balanced_accuracy_perfect():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert balanced_accuracy_score(y_true, y_pred) == 1.0


def test_balanced_accuracy_all_wrong():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 2, 0])
    assert balanced_accuracy_score(y_true, y_pred) == 0.0


def test_balanced_accuracy_single_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert balanced_accuracy_score(y_true, y_pred) == 1.0


def test_balanced_accuracy_imbalanced():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0])
    # Class 0 accuracy = 5/5, class 1 accuracy = 1/2.
    expected = (1.0 + 0.5) / 2.0
    assert np.isclose(balanced_accuracy_score(y_true, y_pred), expected)
