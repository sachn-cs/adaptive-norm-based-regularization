"""Comprehensive tests for penalty values and gradients."""

import numpy as np
import pytest

from regulo.penalty import (
    REGISTRY,
    Covridge,
    ElasticNet,
    Lasso,
    Penalty,
    Ridge,
    Sparridge,
    Void,
)


def gradient_fd(penalty: Penalty, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute penalty gradient via central finite differences."""
    grad = np.zeros_like(weight)
    for r in range(weight.shape[0]):
        for c in range(weight.shape[1]):
            w_plus = weight.copy()
            w_plus[r, c] += eps
            w_minus = weight.copy()
            w_minus[r, c] -= eps
            grad[r, c] = (
                penalty.value(w_plus, 0) - penalty.value(w_minus, 0)
            ) / (2.0 * eps)
    return grad


# Ridge -----------------------------------------------------------------


def test_ridge_valgrad():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    reg = Ridge(lam=0.5)
    assert np.isclose(reg.value(w, 0), 0.5 * np.sum(w**2))
    np.testing.assert_allclose(reg.grad(w, 0), 2.0 * 0.5 * w)


def test_ridge_fd():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3))
    reg = Ridge(lam=0.7)
    np.testing.assert_allclose(reg.grad(w, 0), gradient_fd(reg, w), atol=1e-5)


def test_ridge_zero_weights():
    w = np.zeros((3, 4))
    reg = Ridge(lam=0.1)
    assert reg.value(w, 0) == 0.0
    np.testing.assert_allclose(reg.grad(w, 0), np.zeros_like(w))


def test_ridge_negative_lam_raises():
    with pytest.raises(ValueError):
        Ridge(lam=-0.1)


def test_ridge_large_weights():
    w = np.ones((10, 10)) * 1e6
    reg = Ridge(lam=1.0)
    assert np.isfinite(reg.value(w, 0))
    assert np.all(np.isfinite(reg.grad(w, 0)))


def test_ridge_applies_all():
    assert Ridge(0.1).applies(0)
    assert Ridge(0.1).applies(5)


# Lasso -----------------------------------------------------------------


def test_lasso_valgrad():
    w = np.array([[1.0, -2.0], [0.0, 3.0]])
    reg = Lasso(gamma=0.5)
    assert np.isclose(reg.value(w, 0), 0.5 * np.sum(np.abs(w)))
    np.testing.assert_allclose(reg.grad(w, 0), 0.5 * np.sign(w))


def test_lasso_fd():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3))
    reg = Lasso(gamma=0.3)
    np.testing.assert_allclose(reg.grad(w, 0), gradient_fd(reg, w), atol=1e-5)


def test_lasso_subgrad_zero():
    w = np.zeros((3, 4))
    reg = Lasso(gamma=0.1)
    assert reg.value(w, 0) == 0.0
    np.testing.assert_allclose(reg.grad(w, 0), np.zeros_like(w))


def test_lasso_negative_gamma_raises():
    with pytest.raises(ValueError):
        Lasso(gamma=-0.1)


# ElasticNet ------------------------------------------------------------


def test_elastic_valgrad():
    w = np.array([[1.0, -2.0], [0.0, 3.0]])
    reg = ElasticNet(alpha=0.5, gamma=0.4)
    expected_v = 0.5 * 0.4 * np.sum(np.abs(w)) + 0.5 * 0.5 * np.sum(w**2)
    assert np.isclose(reg.value(w, 0), expected_v)
    expected_g = 0.5 * 0.4 * np.sign(w) + 0.5 * w
    np.testing.assert_allclose(reg.grad(w, 0), expected_g)


def test_elastic_fd():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3))
    reg = ElasticNet(alpha=0.5, gamma=0.2)
    np.testing.assert_allclose(reg.grad(w, 0), gradient_fd(reg, w), atol=1e-5)


def test_elastic_alpha_bounds():
    with pytest.raises(ValueError):
        ElasticNet(alpha=-0.1, gamma=0.1)
    with pytest.raises(ValueError):
        ElasticNet(alpha=1.1, gamma=0.1)


def test_elastic_alpha_zero_half_ridge():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    en = ElasticNet(alpha=0.0, gamma=0.0)
    ridge = Ridge(lam=0.5)
    np.testing.assert_allclose(en.value(w, 0), ridge.value(w, 0))


# Covridge --------------------------------------------------------------


def test_covridge_val():
    w = np.array([[1.0], [2.0]])
    gram = np.array([[2.0, 0.0], [0.0, 3.0]])
    reg = Covridge(lambda1=0.5, lambda2=0.25, gram=gram)
    expected = 0.5 * np.sum((np.sqrt(gram) @ w) ** 2) + 0.25 * np.sum(w**2)
    assert np.isclose(reg.value(w, 0), expected)


def test_covridge_grad():
    w = np.array([[1.0], [2.0]])
    gram = np.array([[2.0, 0.0], [0.0, 3.0]])
    reg = Covridge(lambda1=0.5, lambda2=0.25, gram=gram)
    expected_g = 2.0 * 0.5 * (gram @ w) + 2.0 * 0.25 * w
    np.testing.assert_allclose(reg.grad(w, 0), expected_g)


def test_covridge_fd():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3))
    gram = np.diag([1.0, 2.0, 3.0, 4.0])
    reg = Covridge(lambda1=0.1, lambda2=0.2, gram=gram)
    np.testing.assert_allclose(reg.grad(w, 0), gradient_fd(reg, w), atol=1e-5)


def test_covridge_identity_is_ridge():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    reg = Covridge(lambda1=0.5, lambda2=0.25, gram=np.eye(2))
    ridge = Ridge(lam=0.75)
    np.testing.assert_allclose(reg.value(w, 0), ridge.value(w, 0))
    np.testing.assert_allclose(reg.grad(w, 0), ridge.grad(w, 0))


def test_covridge_applies_layer_zero():
    reg = Covridge(lambda1=0.1, lambda2=0.1, gram=np.eye(2))
    assert reg.applies(0)
    assert not reg.applies(1)
    assert not reg.applies(5)


def test_covridge_negative_lambda_raises():
    with pytest.raises(ValueError):
        Covridge(lambda1=-0.1, lambda2=0.1, gram=np.eye(2))


def test_covridge_zero_weights():
    w = np.zeros((3, 2))
    reg = Covridge(lambda1=0.5, lambda2=0.25, gram=np.eye(3))
    assert reg.value(w, 0) == 0.0
    np.testing.assert_allclose(reg.grad(w, 0), np.zeros_like(w))


def test_covridge_small_eigs():
    gram = np.diag([1e-12, 1e-12])
    w = np.ones((2, 2))
    reg = Covridge(lambda1=0.1, lambda2=0.1, gram=gram)
    assert np.isfinite(reg.value(w, 0))
    assert np.all(np.isfinite(reg.grad(w, 0)))


# Sparridge -------------------------------------------------------------


def test_sparridge_val():
    w = np.array([[1.0], [-2.0]])
    gram = np.array([[1.0, 0.0], [0.0, 1.0]])
    reg = Sparridge(lambda1=0.5, gamma=0.25, gram=gram)
    expected = 0.5 * np.sum(w**2) + 0.25 * np.sum(np.abs(w))
    assert np.isclose(reg.value(w, 0), expected)


def test_sparridge_grad():
    w = np.array([[1.0], [-2.0]])
    gram = np.eye(2)
    reg = Sparridge(lambda1=0.5, gamma=0.25, gram=gram)
    expected_g = 2.0 * 0.5 * w + 0.25 * np.sign(w)
    np.testing.assert_allclose(reg.grad(w, 0), expected_g)


def test_sparridge_fd():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3))
    gram = np.diag([1.0, 2.0, 3.0, 4.0])
    reg = Sparridge(lambda1=0.1, gamma=0.2, gram=gram)
    np.testing.assert_allclose(reg.grad(w, 0), gradient_fd(reg, w), atol=1e-5)


def test_sparridge_identity():
    w = np.array([[1.0, -2.0], [0.0, 3.0]])
    gram = np.eye(2)
    reg = Sparridge(lambda1=0.5, gamma=0.25, gram=gram)
    expected_v = 0.5 * np.sum(w**2) + 0.25 * np.sum(np.abs(w))
    assert np.isclose(reg.value(w, 0), expected_v)


def test_sparridge_applies_layer_zero():
    reg = Sparridge(lambda1=0.1, gamma=0.1, gram=np.eye(2))
    assert reg.applies(0)
    assert not reg.applies(1)


def test_sparridge_negative_raises():
    with pytest.raises(ValueError):
        Sparridge(lambda1=-0.1, gamma=0.1, gram=np.eye(2))
    with pytest.raises(ValueError):
        Sparridge(lambda1=0.1, gamma=-0.1, gram=np.eye(2))


# Void ------------------------------------------------------------------


def test_void_zero():
    w = np.ones((5, 5))
    reg = Void()
    assert reg.value(w, 0) == 0.0
    np.testing.assert_allclose(reg.grad(w, 0), np.zeros_like(w))


def test_void_applies_all():
    assert Void().applies(0)
    assert Void().applies(3)


# Registry & polymorphic dispatch --------------------------------------


def test_registry_complete():
    assert set(REGISTRY) == {"none", "ridge", "lasso", "elastic_net", "covridge", "sparridge"}


def test_each_penalty_has_name():
    for cls in (Void, Ridge, Lasso, ElasticNet, Covridge, Sparridge):
        assert cls.name
        assert isinstance(cls.hp, tuple)


def test_penalty_abstract():
    with pytest.raises(TypeError):
        Penalty()


def test_penalty_repr_default():
    assert "Void" in repr(Void())


def test_sparridge_off_diag_gram():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3))
    gram = np.array(
        [[2.0, 0.5, 0.5, 0.5], [0.5, 2.0, 0.5, 0.5],
         [0.5, 0.5, 2.0, 0.5], [0.5, 0.5, 0.5, 2.0]]
    )
    reg = Sparridge(lambda1=0.3, gamma=0.1, gram=gram)
    cw = reg.csqrt @ w
    expected_v = 0.3 * float(np.sum(cw**2)) + 0.1 * float(np.sum(np.abs(w)))
    assert np.isclose(reg.value(w, 0), expected_v)
