"""Comprehensive tests for loss functions."""

import numpy as np
import pytest

from regulo.loss import Loss, Softmax, Square


def test_loss_abstract():
    with pytest.raises(TypeError):
        Loss()


def test_square_perfect():
    y = np.array([[1.0], [2.0], [3.0]])
    loss_fn = Square()
    assert loss_fn.value(y, y) == 0.0
    np.testing.assert_allclose(loss_fn.grad(y, y), 0.0)


def test_square_known():
    ypred = np.array([[0.0], [2.0]])
    truth = np.array([[1.0], [1.0]])
    loss_fn = Square()
    assert loss_fn.value(ypred, truth) == 1.0


def test_square_grad_shape():
    ypred = np.random.randn(8, 3)
    truth = np.random.randn(8, 3)
    loss_fn = Square()
    grad = loss_fn.grad(ypred, truth)
    assert grad.shape == (8, 3)


def test_square_single():
    ypred = np.array([[5.0]])
    truth = np.array([[3.0]])
    loss_fn = Square()
    assert loss_fn.value(ypred, truth) == 4.0
    np.testing.assert_allclose(loss_fn.grad(ypred, truth), [[4.0]])


def test_softmax_known():
    logits = np.array([[1.0, 0.0]])
    target = np.array([0])
    loss_fn = Softmax()
    loss = loss_fn.value(logits, target)
    expected = float(np.log(1 + np.exp(-1)))
    assert np.isclose(loss, expected)


def test_softmax_perfect():
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


def test_softmax_grad_sum_zero():
    logits = np.random.randn(8, 3)
    target = np.random.randint(0, 3, size=(8,))
    grad = Softmax().grad(logits, target)
    np.testing.assert_allclose(np.sum(grad, axis=1), 0.0, atol=1e-12)


def test_softmax_rejects_float():
    logits = np.array([[1.0, 0.0]])
    target = np.array([0.5])
    loss_fn = Softmax()
    with pytest.raises(TypeError):
        loss_fn.value(logits, target)


def test_softmax_rejects_out_of_range():
    logits = np.array([[1.0, 0.0]])
    target = np.array([5])
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


def test_loss_names():
    assert Softmax().name == "softmax"
    assert Square().name == "square"
