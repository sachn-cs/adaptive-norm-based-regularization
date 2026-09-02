"""Comprehensive tests for loss functions."""

import numpy as np
import pytest

from regulo.loss import Loss, Softmax, Square


def test_loss_is_abstract():
    # Cannot instantiate the abstract base class.
    with pytest.raises(TypeError):
        Loss()


def test_square_perfect_predictions():
    y = np.array([[1.0], [2.0], [3.0]])
    loss_fn = Square()
    assert loss_fn.value(y, y) == 0.0
    np.testing.assert_allclose(loss_fn.grad(y, y), 0.0)


def test_square_known_value():
    y_pred = np.array([[0.0], [2.0]])
    y_true = np.array([[1.0], [1.0]])
    loss_fn = Square()
    # MSE = ((0-1)^2 + (2-1)^2) / 2 = 1.0
    assert loss_fn.value(y_pred, y_true) == 1.0


def test_square_grad_shape():
    y_pred = np.random.randn(8, 3)
    y_true = np.random.randn(8, 3)
    loss_fn = Square()
    grad = loss_fn.grad(y_pred, y_true)
    assert grad.shape == (8, 3)


def test_square_single_sample():
    y_pred = np.array([[5.0]])
    y_true = np.array([[3.0]])
    loss_fn = Square()
    assert loss_fn.value(y_pred, y_true) == 4.0
    np.testing.assert_allclose(loss_fn.grad(y_pred, y_true), [[4.0]])


def test_softmax_known_value():
    logits = np.array([[1.0, 0.0]])
    target = np.array([0])
    loss_fn = Softmax()
    loss = loss_fn.value(logits, target)
    expected = float(np.log(1 + np.exp(-1)))
    assert np.isclose(loss, expected)


def test_softmax_zero_loss_for_perfect_prediction():
    # Very large logit for correct class yields exactly 0.0 loss
    # thanks to the 1e-15 prob floor.
    logits = np.array([[1e6, -1e6, -1e6]])
    target = np.array([0])
    loss_fn = Softmax()
    assert loss_fn.value(logits, target) == 0.0


def test_softmax_grad_shape():
    logits = np.random.randn(8, 3)
    target = np.random.randint(0, 3, size=(8,))
    loss_fn = Softmax()
    grad = loss_fn.grad(logits, target)
    assert grad.shape == (8, 3)


def test_softmax_grad_sums_to_zero_per_row():
    # Closed-form gradient of softmax cross-entropy sums to zero
    # across classes for each sample.
    logits = np.random.randn(8, 3)
    target = np.random.randint(0, 3, size=(8,))
    grad = Softmax().grad(logits, target)
    np.testing.assert_allclose(np.sum(grad, axis=1), 0.0, atol=1e-12)


def test_softmax_rejects_non_integer_target():
    logits = np.array([[1.0, 0.0]])
    target = np.array([0.5])  # float, not integer
    loss_fn = Softmax()
    with pytest.raises(TypeError):
        loss_fn.value(logits, target)


def test_softmax_rejects_out_of_range_target():
    logits = np.array([[1.0, 0.0]])
    target = np.array([5])  # out of range
    loss_fn = Softmax()
    with pytest.raises(ValueError):
        loss_fn.value(logits, target)


def test_softmax_single_class():
    logits = np.array([[0.0], [1.0], [2.0]])
    target = np.array([0, 0, 0])
    loss_fn = Softmax()
    assert np.isclose(loss_fn.value(logits, target), 0.0)
    np.testing.assert_allclose(loss_fn.grad(logits, target), 0.0)


def test_loss_repr():
    assert "Square" in repr(Square())
    assert "Softmax" in repr(Softmax())


def test_softmax_known_value_via_repr():
    # Coverage for the Softmax.name class attribute.
    assert Softmax().name == "softmax"
    assert Square().name == "square"
