"""Comprehensive tests for loss functions."""

import numpy as np

from regulo.losses import CrossEntropyLoss, MSELoss


def test_mse_perfect_predictions():
    y = np.array([[1.0], [2.0], [3.0]])
    loss_fn = MSELoss()
    assert loss_fn.forward(y, y) == 0.0
    np.testing.assert_allclose(loss_fn.backward(y, y), 0.0)


def test_mse_known_value():
    y_pred = np.array([[0.0], [2.0]])
    y_true = np.array([[1.0], [1.0]])
    loss_fn = MSELoss()
    # MSE = ((0-1)^2 + (2-1)^2) / 2 = (1 + 1) / 2 = 1.0
    assert loss_fn.forward(y_pred, y_true) == 1.0


def test_mse_backward_shape():
    y_pred = np.random.randn(8, 3)
    y_true = np.random.randn(8, 3)
    loss_fn = MSELoss()
    grad = loss_fn.backward(y_pred, y_true)
    assert grad.shape == (8, 3)


def test_mse_single_sample():
    y_pred = np.array([[5.0]])
    y_true = np.array([[3.0]])
    loss_fn = MSELoss()
    assert loss_fn.forward(y_pred, y_true) == 4.0
    np.testing.assert_allclose(loss_fn.backward(y_pred, y_true), [[4.0]])


def test_cross_entropy_known_value():
    logits = np.array([[1.0, 0.0]])
    y_true = np.array([0])
    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(logits, y_true)
    # softmax([1,0]) = [e/(e+1), 1/(e+1)]
    # -log(e/(e+1)) = log(1 + 1/e)
    expected = float(np.log(1 + np.exp(-1)))
    assert np.isclose(loss, expected)


def test_cross_entropy_perfect_predictions():
    # Very large logit for correct class.
    logits = np.array([[1e6, -1e6, -1e6]])
    y_true = np.array([0])
    loss_fn = CrossEntropyLoss()
    assert np.isclose(loss_fn.forward(logits, y_true), 0.0, atol=1e-6)


def test_cross_entropy_backward_shape():
    logits = np.random.randn(8, 3)
    y_true = np.random.randint(0, 3, size=(8,))
    loss_fn = CrossEntropyLoss()
    grad = loss_fn.backward(logits, y_true)
    assert grad.shape == (8, 3)


def test_cross_entropy_backward_sum():
    # Gradient should sum to zero across classes for each sample.
    logits = np.random.randn(8, 3)
    y_true = np.random.randint(0, 3, size=(8,))
    grad = CrossEntropyLoss().backward(logits, y_true)
    np.testing.assert_allclose(np.sum(grad, axis=1), 0.0, atol=1e-12)


def test_cross_entropy_single_class():
    # Only one class.
    logits = np.array([[0.0], [1.0], [2.0]])
    y_true = np.array([0, 0, 0])
    loss_fn = CrossEntropyLoss()
    assert np.isclose(loss_fn.forward(logits, y_true), 0.0)
    np.testing.assert_allclose(loss_fn.backward(logits, y_true), 0.0)
