r"""Data generators for simulations and real-data loaders.

Provides synthetic data-generating processes (DGPs) that mirror the
experimental setup of the paper, plus loaders for the two real-world
datasets used in the study.

Synthetic DGPs
--------------
All DGPs share a common mathematical model:

.. math::

    X_{\\text{info}} &\\sim \\mathcal{N}(0, \\Sigma_k) \\quad
        \\text{where } \\Sigma_k = (1-\\rho)I_k + \\rho \\mathbf{1}\\mathbf{1}^T
    \\\\
    X_{\\text{noise}} &\\sim \\mathcal{N}(0, I_{p-k})
    \\\\
    \\theta &\\sim \\mathcal{N}(0, \\tau I_k)
    \\\\
    y &= X_{\\text{info}} \\theta + \\varepsilon
        \\quad\\text{(or } \\sin(X_{\\text{info}}) \\theta + \\varepsilon\\text{)}

where ``epsilon ~ N(0, sigma_noise^2 I)``.

* :func:`make_dgp` -- generic generator with configurable ``n``, ``p``,
  ``k``, correlation ``rho``, noise level, and linear/nonlinear signal.
* :func:`make_dgp1` -- ``n=200, p=20, k=10`` (small).
* :func:`make_dgp2` -- ``n=1000, p=200, k=100`` (medium).
* :func:`make_dgp3` -- ``n=500, p=2000, k=100`` (high-dimensional).

Real-data loaders
-----------------
* :func:`load_energy_data` -- UCI Energy Efficiency (cooling load).
* :func:`load_leukemia_data` -- GSE9476 leukemia microarray with ANOVA
  feature selection.

All loaders return ``(X_train, X_test, y_train, y_test, scaler)`` tuples
with standardised features.  Standardisation is fit on the training split
only to prevent data leakage.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _make_covariance_matrix(k: int, rho: float) -> np.ndarray:
    """Build a ``k x k`` equi-correlation covariance matrix.

    The diagonal is ``1`` and every off-diagonal entry is ``rho``.
    This is the equi-correlation (compound symmetry) structure used in
    the paper's DGP to introduce controlled multicollinearity among
    informative features.

    The matrix has the closed form
    ``Sigma_k = (1 - rho) I_k + rho * 1_k 1_k^T`` and is positive
    definite exactly when ``rho in (-1/(k-1), 1)``.  In particular:

    * ``rho == 0`` recovers the identity covariance -- informative
      features become independent standard normals.
    * ``rho -> 1`` makes informative features perfectly collinear.
    * Negative ``rho`` produces an oscillating correlation structure
      that still satisfies positive-definiteness for moderate
      magnitudes.

    Args:
        k: Dimension of the matrix.  Must be positive.
        rho: Off-diagonal correlation value.  Caller is responsible for
            keeping ``rho`` within the positive-definite range when
            non-trivial.

    Returns:
        Covariance matrix of shape ``(k, k)``.
    """
    sigma = np.full((k, k), rho)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def make_dgp(
    n: int,
    p: int,
    k: int,
    rho: float,
    sigma_noise: float,
    tau: float = 1.0,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data according to the paper's DGP.

    The first ``k`` features are drawn from a multivariate normal with
    equi-correlation ``rho``; the remaining ``p - k`` features are i.i.d.
    standard normal (pure noise).  True coefficients are drawn from
    ``N(0, tau)`` and the response is either linear or sinusoidal.

    The signal-to-noise ratio is controlled by ``tau`` and
    ``sigma_noise``: larger ``tau`` relative to ``sigma_noise`` produces
    a stronger signal.

    Args:
        n: Number of samples.
        p: Total number of features.
        k: Number of informative (correlated) features.
        rho: Pairwise correlation among informative features.
        sigma_noise: Standard deviation of additive Gaussian noise.
        tau: Standard deviation of the true coefficient vector.
        nonlinear: If ``True``, use ``sin(X) @ theta`` instead of
            ``X @ theta``.
        random_state: Seed for the NumPy default RNG.

    Returns:
        ``(X, y)`` with shapes ``(n, p)`` and ``(n, 1)``.
    """
    rng = np.random.default_rng(random_state)
    # Informative block with equi-correlation structure.
    if k > 0:
        cov = _make_covariance_matrix(k, rho)
        x_informative = rng.multivariate_normal(
            mean=np.zeros(k), cov=cov, size=n
        )
    else:
        x_informative = np.empty((n, 0))
    # Noise features: i.i.d. standard normal, uncorrelated with signal.
    x_noise = rng.standard_normal(size=(n, p - k))
    x = np.hstack([x_informative, x_noise])
    # True coefficients.
    theta = rng.normal(0.0, tau, size=k)
    if k == 0:
        y = np.zeros(n)
    elif nonlinear:
        # Sinusoidal signal: y = sum(theta_j * sin(x_j)).
        y = np.sum(theta * np.sin(x_informative), axis=1)
    else:
        # Linear signal: y = X_info @ theta.
        y = np.ravel(x_informative @ theta)
    # Additive Gaussian noise.
    y += rng.normal(0.0, sigma_noise, size=n)
    return x, y.reshape(-1, 1)


def make_dgp1(
    rho: float = 0.25,
    sigma_noise: float = 0.10,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """DGP1: small-scale with ``n=200, p=20, k=10``.

    This is the baseline scenario in Table 1 of the paper.  Half of the
    features are informative and correlated; the other half are noise.

    Args:
        rho: Correlation among informative features.
        sigma_noise: Noise standard deviation.
        nonlinear: Use sinusoidal signal.
        random_state: RNG seed.

    Returns:
        ``(X, y)`` tuple.
    """
    return make_dgp(
        200,
        20,
        10,
        rho,
        sigma_noise,
        nonlinear=nonlinear,
        random_state=random_state,
    )


def make_dgp2(
    rho: float = 0.25,
    sigma_noise: float = 0.10,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """DGP2: medium-scale with ``n=1000, p=200, k=100``.

    Tests the regularizers in a higher-dimensional setting where
    ``p > n`` is not yet true but the feature space is large.

    Args:
        rho: Correlation among informative features.
        sigma_noise: Noise standard deviation.
        nonlinear: Use sinusoidal signal.
        random_state: RNG seed.

    Returns:
        ``(X, y)`` tuple.
    """
    return make_dgp(
        1000,
        200,
        100,
        rho,
        sigma_noise,
        nonlinear=nonlinear,
        random_state=random_state,
    )


def make_dgp3(
    rho: float = 0.25,
    sigma_noise: float = 0.10,
    nonlinear: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """DGP3: high-dimensional with ``n=500, p=2000, k=100``.

    The ``p >> n`` regime where regularization is critical.  Only 5%
    of features are informative.

    Args:
        rho: Correlation among informative features.
        sigma_noise: Noise standard deviation.
        nonlinear: Use sinusoidal signal.
        random_state: RNG seed.

    Returns:
        ``(X, y)`` tuple.
    """
    return make_dgp(
        500,
        2000,
        100,
        rho,
        sigma_noise,
        nonlinear=nonlinear,
        random_state=random_state,
    )


def load_energy_data(
    test_size: float = 0.25,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load the UCI Energy Efficiency dataset (cooling load target ``y2``).

    The dataset has 8 features (relative compactness, surface area, wall
    area, roof area, overall height, orientation, glazing area, glazing
    area distribution) and 2 targets (heating load ``y1``, cooling load
    ``y2``).  This loader returns ``y2`` as the target, matching the
    paper's choice.

    Standardization
    ---------------
    Features are z-scored using :class:`StandardScaler`.  The scaler
    is **fit on the training split only** and then applied to the test
    split, preventing information leakage from test into train.  The
    fitted scaler is returned so callers can apply it to additional
    held-out data (e.g. a validation set collected from the same
    population).

    Args:
        test_size: Fraction of samples reserved for the test set.
        random_state: Seed for the train/test split.

    Returns:
        ``(X_train, X_test, y_train, y_test, scaler)``:

        * ``X_train`` -- standardized training features, shape
          ``(n_train, 8)``.
        * ``X_test`` -- standardized test features, shape
          ``(n_test, 8)``.
        * ``y_train`` -- training targets, shape ``(n_train, 1)``.
        * ``y_test`` -- test targets, shape ``(n_test, 1)``.
        * ``scaler`` -- the fitted :class:`StandardScaler` for reuse
          on additional data.

    Raises:
        sklearn.datasets._fetch_fail: Surfaced from OpenML if the
            dataset is unavailable; this loader does not retry or
            cache.
    """
    data = fetch_openml(name="energy-efficiency", as_frame=True, parser="auto")
    df = data.frame
    # Targets: 'y1' (heating), 'y2' (cooling).  We use y2.
    feature_cols = [c for c in df.columns if c not in ("y1", "y2")]
    x = df[feature_cols].astype(float).to_numpy()
    y = df["y2"].astype(float).to_numpy().reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler


def load_leukemia_data(
    n_features: int = 2000,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load the GSE9476 leukemia microarray dataset.

    ANOVA feature selection (``f_classif``) reduces the feature space to
    the top *n_features* most discriminative probes.  Features are then
    standardised on the training split.

    The train/test split uses stratification (``stratify=y``) to preserve
    class proportions, which is important for the imbalanced classes in
    this dataset.

    Args:
        n_features: Number of features to keep via ANOVA F-test.  Must
            be ``<= total probes``; the loader does not validate and
            ``numpy.argsort`` will silently truncate if the value
            exceeds the available count.
        test_size: Fraction of samples reserved for the test set.
        random_state: Seed for the train/test split.

    Returns:
        ``(X_train, X_test, y_train, y_test, scaler)``:

        * ``X_train`` -- standardized training features, shape
          ``(n_train, n_features)``.
        * ``X_test`` -- standardized test features, shape
          ``(n_test, n_features)``.
        * ``y_train`` -- integer-encoded training labels in
          ``[0, n_classes)``, shape ``(n_train,)``.
        * ``y_test`` -- integer-encoded test labels, shape
          ``(n_test,)``.
        * ``scaler`` -- the fitted :class:`StandardScaler` for reuse
          on additional data.

    Raises:
        RuntimeError: If the dataset cannot be fetched from OpenML.
            The original exception is chained via ``raise ... from``.
    """
    # Attempt to fetch from OpenML (GSE9476 surrogate).  Catching
    # ``Exception`` is deliberate: sklearn raises different concrete
    # subclasses depending on the failure mode (network, schema,
    # parse), and we want a single, actionable error for callers.
    try:
        data = fetch_openml(data_id=1120, as_frame=True, parser="auto")
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch GSE9476 from OpenML. "
            "Please download manually and place in data/gse9476.csv."
        ) from exc
    df = data.frame
    y_raw = df[data.target_names[0]].to_numpy()
    x = df.drop(columns=data.target_names).astype(float).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # ANOVA feature selection: rank features by F-statistic (between-
    # group variance / within-group variance) and keep the top
    # ``n_features``.  This reduces the dimensionality from tens of
    # thousands of probes to a manageable size while preserving the
    # most class-discriminative signal.
    f_values, _ = f_classif(x, y)
    top_idx = np.argsort(f_values)[-n_features:]
    x = x[:, top_idx]
    # Stratified split to preserve class balance across train/test.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler
