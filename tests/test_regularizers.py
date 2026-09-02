"""Comprehensive tests for regularizer penalty values and gradients."""

import numpy as np
import pytest

from regulo.regularizers import (
    Covridge,
    ElasticNet,
    Lasso,
    NoRegularizer,
    Ridge,
    Sparridge,
)

# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------


def test_ridge_penalty_and_gradient():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    reg = Ridge(lambda_=0.5)
    expected_penalty = 0.5 * np.sum(w**2)
    assert np.isclose(reg.penalty(w), expected_penalty)
    expected_grad = 2.0 * 0.5 * w
    np.testing.assert_allclose(reg.gradient(w), expected_grad)


def test_ridge_zero_weights():
    w = np.zeros((3, 4))
    reg = Ridge(lambda_=0.1)
    assert reg.penalty(w) == 0.0
    np.testing.assert_allclose(reg.gradient(w), np.zeros_like(w))


def test_ridge_negative_lambda_raises():
    with pytest.raises(ValueError):
        Ridge(lambda_=-0.1)


def test_ridge_large_weights():
    w = np.ones((10, 10)) * 1e6
    reg = Ridge(lambda_=1.0)
    assert np.isfinite(reg.penalty(w))
    grad = reg.gradient(w)
    assert np.all(np.isfinite(grad))


def test_ridge_non_square_weights():
    w = np.ones((5, 3))
    reg = Ridge(lambda_=0.5)
    assert reg.penalty(w) == 0.5 * 15.0
    np.testing.assert_allclose(reg.gradient(w), w)


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------


def test_lasso_penalty_and_gradient():
    w = np.array([[1.0, -2.0], [0.0, 3.0]])
    reg = Lasso(gamma=0.5)
    expected_penalty = 0.5 * np.sum(np.abs(w))
    assert np.isclose(reg.penalty(w), expected_penalty)
    expected_grad = 0.5 * np.sign(w)
    np.testing.assert_allclose(reg.gradient(w), expected_grad)


def test_lasso_zero_weights():
    w = np.zeros((3, 4))
    reg = Lasso(gamma=0.1)
    assert reg.penalty(w) == 0.0
    np.testing.assert_allclose(reg.gradient(w), np.zeros_like(w))


def test_lasso_negative_gamma_raises():
    with pytest.raises(ValueError):
        Lasso(gamma=-0.1)


def test_lasso_subgradient_at_zero():
    w = np.array([[0.0, 0.0]])
    reg = Lasso(gamma=1.0)
    grad = reg.gradient(w)
    assert np.all(grad == 0.0)


# ---------------------------------------------------------------------------
# Elastic Net
# ---------------------------------------------------------------------------


def test_elastic_net_penalty_and_gradient():
    w = np.array([[1.0, -2.0], [0.0, 3.0]])
    reg = ElasticNet(alpha=0.5, gamma=1.0)
    expected_penalty = 0.5 * 1.0 * np.sum(np.abs(w)) + (1 - 0.5) * 0.5 * np.sum(
        w**2
    )
    assert np.isclose(reg.penalty(w), expected_penalty)
    expected_grad = 0.5 * 1.0 * np.sign(w) + (1 - 0.5) * w
    np.testing.assert_allclose(reg.gradient(w), expected_grad)


def test_elastic_net_alpha_bounds():
    with pytest.raises(ValueError):
        ElasticNet(alpha=-0.1, gamma=1.0)
    with pytest.raises(ValueError):
        ElasticNet(alpha=1.1, gamma=1.0)


def test_elastic_net_zero_alpha_is_ridge():
    # When alpha=0, ElasticNet = (1/2) * ‖W‖_F^2, equivalent to Ridge(0.5).
    w = np.array([[1.0, 2.0]])
    reg = ElasticNet(alpha=0.0, gamma=1.0)
    ridge = Ridge(lambda_=0.5)
    np.testing.assert_allclose(reg.penalty(w), ridge.penalty(w))
    np.testing.assert_allclose(reg.gradient(w), ridge.gradient(w))


def test_elastic_net_alpha_one_is_lasso():
    w = np.array([[1.0, -2.0]])
    reg = ElasticNet(alpha=1.0, gamma=0.5)
    lasso = Lasso(gamma=0.5)
    np.testing.assert_allclose(reg.penalty(w), lasso.penalty(w))
    np.testing.assert_allclose(reg.gradient(w), lasso.gradient(w))


# ---------------------------------------------------------------------------
# Covridge
# ---------------------------------------------------------------------------


def test_covridge_penalty():
    w = np.array([[1.0], [2.0]])
    c = np.array([[2.0, 0.0], [0.0, 3.0]])
    reg = Covridge(lambda1=0.5, lambda2=0.25, c_delta_n=c)
    cw = np.sqrt(c) @ w
    expected = 0.5 * np.sum(cw**2) + 0.25 * np.sum(w**2)
    assert np.isclose(reg.penalty(w), expected)


def test_covridge_gradient():
    w = np.array([[1.0], [2.0]])
    c = np.array([[2.0, 0.0], [0.0, 3.0]])
    reg = Covridge(lambda1=0.5, lambda2=0.25, c_delta_n=c)
    expected_grad = 2.0 * 0.5 * (c @ w) + 2.0 * 0.25 * w
    np.testing.assert_allclose(reg.gradient(w), expected_grad)


def test_covridge_identity_c():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    c = np.eye(2)
    reg = Covridge(lambda1=0.5, lambda2=0.25, c_delta_n=c)
    # With C=I, Covridge reduces to Ridge(lambda1+lambda2).
    ridge = Ridge(lambda_=0.75)
    np.testing.assert_allclose(reg.penalty(w), ridge.penalty(w))
    np.testing.assert_allclose(reg.gradient(w), ridge.gradient(w))


def test_covridge_negative_lambda_raises():
    with pytest.raises(ValueError):
        Covridge(lambda1=-0.1, lambda2=0.1, c_delta_n=np.eye(2))


def test_covridge_zero_weights():
    w = np.zeros((3, 2))
    c = np.eye(3)
    reg = Covridge(lambda1=0.5, lambda2=0.25, c_delta_n=c)
    assert reg.penalty(w) == 0.0
    np.testing.assert_allclose(reg.gradient(w), np.zeros_like(w))


def test_covridge_numerical_stability_small_eigenvalues():
    # C with tiny eigenvalues to test stability.
    c = np.diag([1e-12, 1e-12])
    w = np.ones((2, 2))
    reg = Covridge(lambda1=0.1, lambda2=0.1, c_delta_n=c)
    assert np.isfinite(reg.penalty(w))
    grad = reg.gradient(w)
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# Sparridge
# ---------------------------------------------------------------------------


def test_sparridge_penalty():
    w = np.array([[1.0], [-2.0]])
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    reg = Sparridge(lambda1=0.5, gamma=0.25, c_delta_n=c)
    expected = 0.5 * np.sum(w**2) + 0.25 * np.sum(np.abs(w))
    assert np.isclose(reg.penalty(w), expected)


def test_sparridge_gradient():
    w = np.array([[1.0], [-2.0]])
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    reg = Sparridge(lambda1=0.5, gamma=0.25, c_delta_n=c)
    expected_grad = 2.0 * 0.5 * w + 0.25 * np.sign(w)
    np.testing.assert_allclose(reg.gradient(w), expected_grad)


def test_sparridge_identity_c_reduces_to_en():
    w = np.array([[1.0, -2.0], [0.0, 3.0]])
    c = np.eye(2)
    reg = Sparridge(lambda1=0.5, gamma=0.25, c_delta_n=c)
    # With C=I, Sparridge = λ1 ‖W‖_F^2 + γ ‖W‖_1.
    # This is ElasticNet(alpha=1, gamma) with extra λ1 ridge.
    expected_penalty = 0.5 * np.sum(w**2) + 0.25 * np.sum(np.abs(w))
    assert np.isclose(reg.penalty(w), expected_penalty)


def test_sparridge_negative_params_raises():
    with pytest.raises(ValueError):
        Sparridge(lambda1=-0.1, gamma=0.1, c_delta_n=np.eye(2))
    with pytest.raises(ValueError):
        Sparridge(lambda1=0.1, gamma=-0.1, c_delta_n=np.eye(2))


def test_sparridge_zero_weights():
    w = np.zeros((3, 2))
    c = np.eye(3)
    reg = Sparridge(lambda1=0.5, gamma=0.25, c_delta_n=c)
    assert reg.penalty(w) == 0.0
    np.testing.assert_allclose(reg.gradient(w), np.zeros_like(w))


# ---------------------------------------------------------------------------
# NoRegularizer
# ---------------------------------------------------------------------------


def test_no_regularizer():
    w = np.ones((5, 5))
    reg = NoRegularizer()
    assert reg.penalty(w) == 0.0
    np.testing.assert_allclose(reg.gradient(w), np.zeros_like(w))
