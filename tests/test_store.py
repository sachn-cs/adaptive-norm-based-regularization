"""Tests for save/load round-trips and version validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from regulo import __version__
from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square
from regulo.net import MLP
from regulo.penalty import Ridge, Void
from regulo.store import load, meta, save, snapshot


def test_meta_fields():
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    m = meta(runner)
    assert m["version"] == __version__
    assert m["shape"] == [3, 4, 1]
    assert m["loss"] == "square"
    assert m["penalty"] == "none"
    assert m["task"] == "regression"
    assert m["adam"]["lr"] == 1e-3


def test_save_load_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lam=0.01),
        Adam(lr=1e-2),
        epochs=3,
    )
    runner.fit(x, y, seed=0)
    weights_before = [w.copy() for w in runner.mlp.weights]
    biases_before = [b.copy() for b in runner.mlp.biases]
    clock_before = runner.adam.clock

    target = tmp_path / "model"
    save(runner, str(target))
    assert (target / "meta.json").exists()
    assert (target / "weights.npz").exists()
    assert (target / "biases.npz").exists()
    assert (target / "adam.npz").exists()

    loaded = load(str(target))
    assert loaded.mlp.shape == [3, 4, 1]
    for wn, wo in zip(loaded.mlp.weights, weights_before):
        np.testing.assert_array_equal(wn, wo)
    for bn, bo in zip(loaded.mlp.biases, biases_before):
        np.testing.assert_array_equal(bn, bo)
    assert loaded.adam.clock == clock_before
    assert loaded.adam.lr == 1e-2


def test_snapshot():
    runner = Runner(MLP([2, 3, 1]), Square(), Void(), Adam())
    target = Path("/tmp/_regulo_snapshot_test")
    save(runner, str(target))
    try:
        m = snapshot(runner)
        assert m["shape"] == [2, 3, 1]
    finally:
        import shutil
        shutil.rmtree(target, ignore_errors=True)


def test_load_rejects_version_mismatch(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "model"
    save(runner, str(target))

    # Tamper with the version field.
    meta_path = target / "meta.json"
    data = json.loads(meta_path.read_text())
    data["version"] = "999.0.0"
    meta_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="version mismatch"):
        load(str(target))


def test_restart_preserves_optimizer(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 3))
    y = rng.standard_normal((16, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lam=0.001),
        Adam(lr=1e-2),
        epochs=2,
    )
    runner.fit(x, y, seed=0)

    target = tmp_path / "warm"
    save(runner, str(target))

    new_runner = Runner(
        MLP([3, 4, 1], seed=1),
        Square(),
        Ridge(lam=0.001),
        Adam(lr=1e-3),
        epochs=2,
    )
    new_runner.restart(str(target))
    np.testing.assert_allclose(
        new_runner.mlp.weights[0], runner.mlp.weights[0]
    )
    assert new_runner.adam.clock == 0
    assert new_runner.adam.lr == 1e-3


def test_restart_rejects_mismatch(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "m"
    save(runner, str(target))
    new_runner = Runner(MLP([3, 8, 1]), Square(), Void(), Adam())
    with pytest.raises(ValueError, match="Architecture mismatch"):
        new_runner.restart(str(target))


def test_load_missing_adam(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "no_adam"
    save(runner, str(target))
    (target / "adam.npz").unlink()
    loaded = load(str(target))
    assert loaded.adam.clock == 0
    for group in loaded.adam.mean:
        for buf in loaded.adam.mean[group]:
            assert buf is None
