"""Comprehensive tests for synthetic DGP generation."""

import numpy as np

from regulo.data import make_dgp, make_dgp1, make_dgp2, make_dgp3


def test_dgp_shape():
    x, y = make_dgp1()
    assert x.shape == (200, 20)
    assert y.shape == (200, 1)


def test_dgp2_shape():
    x, y = make_dgp2()
    assert x.shape == (1000, 200)
    assert y.shape == (1000, 1)


def test_dgp3_shape():
    x, y = make_dgp3()
    assert x.shape == (500, 2000)
    assert y.shape == (500, 1)


def test_dgp_correlation():
    rho = 0.75
    x, _ = make_dgp(500, 20, 10, rho, 0.1, random_state=42)
    # Check approximate correlation in first 10 features.
    emp_corr = np.corrcoef(x[:, :10], rowvar=False)
    off_diag = emp_corr[np.triu_indices_from(emp_corr, k=1)]
    assert np.mean(off_diag) > 0.5


def test_dgp_nonlinear():
    x, y_lin = make_dgp(100, 10, 5, 0.25, 0.1, nonlinear=False, random_state=1)
    _, y_non = make_dgp(100, 10, 5, 0.25, 0.1, nonlinear=True, random_state=1)
    assert not np.allclose(y_lin, y_non)


def test_dgp_zero_correlation():
    x, _ = make_dgp(500, 20, 10, 0.0, 0.1, random_state=42)
    emp_corr = np.corrcoef(x[:, :10], rowvar=False)
    off_diag = emp_corr[np.triu_indices_from(emp_corr, k=1)]
    # Off-diagonal should be close to zero.
    assert np.mean(np.abs(off_diag)) < 0.1


def test_dgp_high_correlation():
    x, _ = make_dgp(500, 20, 10, 0.95, 0.1, random_state=42)
    emp_corr = np.corrcoef(x[:, :10], rowvar=False)
    off_diag = emp_corr[np.triu_indices_from(emp_corr, k=1)]
    assert np.mean(off_diag) > 0.85


def test_dgp_zero_noise():
    x, y = make_dgp(100, 10, 5, 0.25, 0.0, random_state=42)
    # With zero noise and linear signal, y should be deterministic.
    _, y2 = make_dgp(100, 10, 5, 0.25, 0.0, random_state=42)
    np.testing.assert_allclose(y, y2)


def test_dgp_all_informative():
    # k = p, all features informative.
    x, y = make_dgp(50, 10, 10, 0.25, 0.1, random_state=42)
    assert x.shape == (50, 10)


def test_dgp_no_informative():
    # k = 0, all features are noise.
    x, y = make_dgp(50, 10, 0, 0.25, 0.1, random_state=42)
    assert x.shape == (50, 10)
    # y should be pure noise.
    assert np.std(y) > 0.0


def test_dgp_different_seeds():
    x1, y1 = make_dgp(100, 10, 5, 0.25, 0.1, random_state=1)
    x2, y2 = make_dgp(100, 10, 5, 0.25, 0.1, random_state=2)
    assert not np.allclose(x1, x2)
    assert not np.allclose(y1, y2)
