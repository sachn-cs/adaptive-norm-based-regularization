"""Comprehensive tests for the Adam optimizer."""

import numpy as np
import pytest

from regulo.adam import Adam


def test_adam_updates_weights():
    opt = Adam(learning_rate=1e-3)
    params = {"weights": [np.ones((2, 2))]}
    grads = {"weights": [np.ones((2, 2))]}
    new_params = opt.step(params, grads)
    assert not np.array_equal(new_params["weights"][0], params["weights"][0])


def test_adam_moment_estimates():
    opt = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
    params = {"weights": [np.zeros((1, 1))]}
    grads = {"weights": [np.ones((1, 1))]}
    for _ in range(1000):
        params = opt.step(params, grads)
    np.testing.assert_allclose(
        params["weights"][0], np.full((1, 1), -100.0), atol=1e-3
    )


def test_adam_reset():
    opt = Adam()
    params = {"weights": [np.ones((2, 2))]}
    grads = {"weights": [np.ones((2, 2))]}
    opt.step(params, grads)
    assert opt.clock == 1
    opt.reset()
    assert opt.clock == 0
    assert opt.mean == {}
    assert opt.variance == {}


def test_adam_zero_gradient():
    opt = Adam(learning_rate=1e-3)
    params = {"weights": [np.ones((2, 2))]}
    grads = {"weights": [np.zeros((2, 2))]}
    new_params = opt.step(params, grads)
    np.testing.assert_allclose(new_params["weights"][0], params["weights"][0])


def test_adam_large_gradient():
    opt = Adam(learning_rate=1e-3)
    params = {"weights": [np.zeros((1, 1))]}
    grads = {"weights": [np.full((1, 1), 1e6)]}
    new_params = opt.step(params, grads)
    assert np.isfinite(new_params["weights"][0])


def test_adam_multiple_parameter_groups():
    opt = Adam(learning_rate=1e-3)
    params = {
        "weights": [np.ones((2, 2)), np.ones((2, 1))],
        "biases": [np.zeros((1, 2)), np.zeros((1, 1))],
    }
    grads = {
        "weights": [np.ones((2, 2)), np.ones((2, 1))],
        "biases": [np.ones((1, 2)), np.ones((1, 1))],
    }
    new_params = opt.step(params, grads)
    assert len(new_params["weights"]) == 2
    assert len(new_params["biases"]) == 2
    for i in range(2):
        assert not np.array_equal(
            new_params["weights"][i], params["weights"][i]
        )
        assert not np.array_equal(
            new_params["biases"][i], params["biases"][i]
        )


def test_adam_rejects_invalid_beta():
    with pytest.raises(ValueError):
        Adam(beta1=1.0)
    with pytest.raises(ValueError):
        Adam(beta2=1.5)
    with pytest.raises(ValueError):
        Adam(beta1=-0.1)


def test_adam_rejects_non_positive_lr():
    with pytest.raises(ValueError):
        Adam(learning_rate=0.0)
    with pytest.raises(ValueError):
        Adam(learning_rate=-1.0)


def test_adam_rejects_non_positive_epsilon():
    with pytest.raises(ValueError):
        Adam(epsilon=0.0)
    with pytest.raises(ValueError):
        Adam(epsilon=-1e-8)


def test_adam_repr_contains_key_params():
    opt = Adam(learning_rate=0.01, beta1=0.5)
    text = repr(opt)
    assert "learning_rate=0.01" in text
    assert "beta1=0.5" in text
