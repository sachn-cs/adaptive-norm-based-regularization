"""Comprehensive tests for synthetic DGP generation."""

import numpy as np
import pytest

from regulo.data import equicorr, synth


def test_synth_shape_small():
    x, y = synth(200, 20, 10, 0.25, 0.10, seed=0)
    assert x.shape == (200, 20)
    assert y.shape == (200, 1)


def test_synth_shape_mid():
    x, y = synth(1000, 200, 100, 0.25, 0.10, seed=0)
    assert x.shape == (1000, 200)
    assert y.shape == (1000, 1)


def test_synth_shape_wide():
    x, y = synth(500, 2000, 100, 0.25, 0.10, seed=0)
    assert x.shape == (500, 2000)
    assert y.shape == (500, 1)


def test_synth_correlation():
    rho = 0.75
    x, _ = synth(500, 20, 10, rho, 0.1, seed=42)
    emp_corr = np.corrcoef(x[:, :10], rowvar=False)
    off_diag = emp_corr[np.triu_indices_from(emp_corr, k=1)]
    assert np.mean(off_diag) > 0.5


def test_synth_nonlinear():
    x, y_lin = synth(100, 10, 5, 0.25, 0.1, nonlinear=False, seed=1)
    _, y_non = synth(100, 10, 5, 0.25, 0.1, nonlinear=True, seed=1)
    assert not np.allclose(y_lin, y_non)


def test_synth_zero_correlation():
    x, _ = synth(500, 20, 10, 0.0, 0.1, seed=42)
    emp_corr = np.corrcoef(x[:, :10], rowvar=False)
    off_diag = emp_corr[np.triu_indices_from(emp_corr, k=1)]
    assert np.mean(np.abs(off_diag)) < 0.1


def test_synth_high_correlation():
    x, _ = synth(500, 20, 10, 0.95, 0.1, seed=42)
    emp_corr = np.corrcoef(x[:, :10], rowvar=False)
    off_diag = emp_corr[np.triu_indices_from(emp_corr, k=1)]
    assert np.mean(off_diag) > 0.85


def test_synth_zero_noise():
    x, y = synth(100, 10, 5, 0.25, 0.0, seed=42)
    _, y2 = synth(100, 10, 5, 0.25, 0.0, seed=42)
    np.testing.assert_allclose(y, y2)


def test_synth_all_informative():
    x, y = synth(50, 10, 10, 0.25, 0.1, seed=42)
    assert x.shape == (50, 10)


def test_synth_no_informative():
    x, y = synth(50, 10, 0, 0.25, 0.1, seed=42)
    assert x.shape == (50, 10)
    assert np.std(y) > 0.0


def test_synth_different_seeds():
    x1, y1 = synth(100, 10, 5, 0.25, 0.1, seed=1)
    x2, y2 = synth(100, 10, 5, 0.25, 0.1, seed=2)
    assert not np.allclose(x1, x2)
    assert not np.allclose(y1, y2)


def test_synth_rejects_k_gt_p():
    with pytest.raises(ValueError):
        synth(50, 5, 10, 0.25, 0.1)


def test_synth_rejects_neg_n():
    with pytest.raises(ValueError):
        synth(-1, 5, 2, 0.0, 0.1)


def test_synth_rejects_neg_p():
    with pytest.raises(ValueError):
        synth(50, -1, 2, 0.0, 0.1)


def test_synth_rejects_neg_sigma():
    with pytest.raises(ValueError):
        synth(50, 5, 2, 0.0, -0.1)


def test_synth_rejects_neg_tau():
    with pytest.raises(ValueError):
        synth(50, 5, 2, 0.0, 0.1, tau=-1.0)


def test_synth_rejects_non_pd_rho():
    with pytest.raises(ValueError):
        synth(50, 10, 5, 1.0, 0.1)


def test_equicorr_shape():
    sigma = equicorr(5, 0.3)
    assert sigma.shape == (5, 5)
    np.testing.assert_allclose(np.diag(sigma), 1.0)
    sigma_off = sigma.copy()
    np.fill_diagonal(sigma_off, 0.3)
    np.testing.assert_allclose(sigma_off, 0.3)


def test_equicorr_k_one():
    sigma = equicorr(1, 5.0)
    np.testing.assert_allclose(sigma, [[1.0]])


def test_equicorr_rejects_non_pd():
    with pytest.raises(ValueError):
        equicorr(3, 1.0)
    with pytest.raises(ValueError):
        equicorr(3, -1.0)


def test_equicorr_rejects_non_positive_k():
    with pytest.raises(ValueError):
        equicorr(0, 0.0)
    with pytest.raises(ValueError):
        equicorr(-1, 0.0)
