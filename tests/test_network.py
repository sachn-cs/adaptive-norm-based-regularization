"""Comprehensive tests for network forward pass, backward pass, and gradient checks."""

import numpy as np

from regulo.network import FullyConnectedNetwork


def test_forward_shape():
    net = FullyConnectedNetwork([10, 64, 32, 1])
    x = np.random.randn(8, 10)
    y = net.forward(x)
    assert y.shape == (8, 1)


def test_forward_multi_output():
    net = FullyConnectedNetwork([5, 4, 3])
    x = np.random.randn(10, 5)
    y = net.forward(x)
    assert y.shape == (10, 3)


def test_forward_single_sample():
    net = FullyConnectedNetwork([4, 3, 2])
    x = np.random.randn(1, 4)
    y = net.forward(x)
    assert y.shape == (1, 2)


def test_forward_single_layer():
    # No hidden layers.
    net = FullyConnectedNetwork([3, 1])
    x = np.random.randn(5, 3)
    y = net.forward(x)
    assert y.shape == (5, 1)


def test_forward_zero_input():
    net = FullyConnectedNetwork([3, 2, 1])
    x = np.zeros((4, 3))
    y = net.forward(x)
    assert y.shape == (4, 1)
    # With zero input and bias=0, output should be zero.
    np.testing.assert_allclose(y, 0.0, atol=1e-12)


def test_backward_shape():
    net = FullyConnectedNetwork([4, 3, 2])
    x = np.random.randn(16, 4)
    out = net.forward(x)
    dloss = np.ones_like(out)
    grads = net.backward(dloss)
    assert len(grads["weights"]) == 2
    assert len(grads["biases"]) == 2
    assert grads["weights"][0].shape == (4, 3)
    assert grads["weights"][1].shape == (3, 2)
    assert grads["biases"][0].shape == (1, 3)
    assert grads["biases"][1].shape == (1, 2)


def test_backward_single_sample():
    net = FullyConnectedNetwork([3, 4, 1])
    x = np.random.randn(1, 3)
    out = net.forward(x)
    dloss = np.ones_like(out)
    grads = net.backward(dloss)
    assert grads["weights"][0].shape == (3, 4)
    assert grads["weights"][1].shape == (4, 1)


def test_backward_single_layer():
    net = FullyConnectedNetwork([3, 1])
    x = np.random.randn(8, 3)
    out = net.forward(x)
    dloss = np.ones_like(out)
    grads = net.backward(dloss)
    assert len(grads["weights"]) == 1
    assert len(grads["biases"]) == 1
    assert grads["weights"][0].shape == (3, 1)


def test_backward_zero_input():
    net = FullyConnectedNetwork([3, 2, 1])
    x = np.zeros((4, 3))
    out = net.forward(x)
    dloss = np.ones_like(out)
    grads = net.backward(dloss)
    # With zero input, first-layer weight gradient should be zero.
    np.testing.assert_allclose(grads["weights"][0], 0.0, atol=1e-12)


def test_get_set_params():
    net = FullyConnectedNetwork([3, 4, 2])
    params = net.get_params()
    # Modify params.
    params["weights"][0] = np.ones_like(params["weights"][0])
    net.set_params(params)
    new_params = net.get_params()
    np.testing.assert_allclose(
        new_params["weights"][0], np.ones_like(params["weights"][0])
    )


def _finite_difference_gradient(
    net: FullyConnectedNetwork, x: np.ndarray, y: np.ndarray, eps: float = 1e-5
) -> dict:
    """Compute gradients via finite differences for weights."""
    out = net.forward(x)
    loss = float(np.mean((out - y) ** 2))
    fd_grads: dict = {"weights": [], "biases": []}
    for i, w in enumerate(net.weights):
        grad = np.zeros_like(w)
        for row in range(w.shape[0]):
            for col in range(w.shape[1]):
                w_plus = w.copy()
                w_plus[row, col] += eps
                orig = net.weights[i].copy()
                net.weights[i] = w_plus
                out_plus = net.forward(x)
                loss_plus = float(np.mean((out_plus - y) ** 2))
                net.weights[i] = orig
                grad[row, col] = (loss_plus - loss) / eps
        fd_grads["weights"].append(grad)
    for i, b in enumerate(net.biases):
        grad = np.zeros_like(b)
        for col in range(b.shape[1]):
            b_plus = b.copy()
            b_plus[0, col] += eps
            orig = net.biases[i].copy()
            net.biases[i] = b_plus
            out_plus = net.forward(x)
            loss_plus = float(np.mean((out_plus - y) ** 2))
            net.biases[i] = orig
            grad[0, col] = (loss_plus - loss) / eps
        fd_grads["biases"].append(grad)
    return fd_grads


def test_gradient_finite_difference():
    np.random.seed(0)
    net = FullyConnectedNetwork([3, 4, 1])
    x = np.random.randn(4, 3)
    y = np.random.randn(4, 1)
    out = net.forward(x)
    dloss = 2.0 * (out - y) / out.size
    grads = net.backward(dloss)
    fd_grads = _finite_difference_gradient(net, x, y)
    for i in range(len(grads["weights"])):
        np.testing.assert_allclose(
            grads["weights"][i], fd_grads["weights"][i], rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            grads["biases"][i], fd_grads["biases"][i], rtol=1e-3, atol=1e-4
        )


def test_gradient_finite_difference_multi_output():
    np.random.seed(2)
    net = FullyConnectedNetwork([3, 4, 2])
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    out = net.forward(x)
    dloss = 2.0 * (out - y) / out.size
    grads = net.backward(dloss)
    fd_grads = _finite_difference_gradient(net, x, y)
    for i in range(len(grads["weights"])):
        np.testing.assert_allclose(
            grads["weights"][i], fd_grads["weights"][i], rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            grads["biases"][i], fd_grads["biases"][i], rtol=1e-3, atol=1e-4
        )
