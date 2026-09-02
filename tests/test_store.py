"""Tests for save/load round-trips and version validation."""

from pathlib import Path

import numpy as np
import pytest

from regulo import __version__
from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square
from regulo.net import MLP
from regulo.penalty import Ridge, Void
from regulo.store import load, load_meta, meta, save


def test_meta_contains_required_fields():
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    m = meta(runner)
    assert m["version"] == __version__
    assert m["layer_sizes"] == [3, 4, 1]
    assert m["loss"] == "square"
    assert m["penalty"] == "none"
    assert m["task"] == "regression"
    assert m["adam"]["learning_rate"] == 1e-3


def test_save_load_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lambda_=0.01),
        Adam(learning_rate=1e-2),
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
    assert loaded.mlp.layer_sizes == [3, 4, 1]
    for w_new, w_old in zip(loaded.mlp.weights, weights_before):
        np.testing.assert_array_equal(w_new, w_old)
    for b_new, b_old in zip(loaded.mlp.biases, biases_before):
        np.testing.assert_array_equal(b_new, b_old)
    assert loaded.adam.clock == clock_before
    assert loaded.adam.learning_rate == 1e-2


def test_load_meta_returns_dict():
    runner = Runner(MLP([2, 3, 1]), Square(), Void(), Adam())
    target = Path("/tmp/_regulo_meta_test")
    save(runner, str(target))
    try:
        m = load_meta(str(target))
        assert m["layer_sizes"] == [2, 3, 1]
    finally:
        for ext in ("", ".meta.json", "/meta.json"):
            pass
        import shutil

        shutil.rmtree(target, ignore_errors=True)


def test_load_rejects_major_version_mismatch(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "model"
    save(runner, str(target))

    # Tamper with the version field.
    import json

    meta_path = target / "meta.json"
    data = json.loads(meta_path.read_text())
    data["version"] = "999.0.0"
    meta_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="version mismatch"):
        load(str(target))


def test_warm_start_preserves_optimizer_state(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 3))
    y = rng.standard_normal((16, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lambda_=0.001),
        Adam(learning_rate=1e-2),
        epochs=2,
    )
    runner.fit(x, y, seed=0)
    adam_clock_before = runner.adam.clock

    target = tmp_path / "warm"
    save(runner, str(target))

    # New runner, fresh optimizer, same architecture.
    new_runner = Runner(
        MLP([3, 4, 1], seed=1),
        Square(),
        Ridge(lambda_=0.001),
        Adam(learning_rate=1e-3),
        epochs=2,
    )
    new_runner.warm_start(str(target))
    # The weights from disk should have overwritten the random init.
    np.testing.assert_allclose(
        new_runner.mlp.weights[0], runner.mlp.weights[0]
    )
    # Optimizer state untouched by warm_start.
    assert new_runner.adam.clock == 0
    assert new_runner.adam.learning_rate == 1e-3


def test_warm_start_rejects_architecture_mismatch(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "m"
    save(runner, str(target))
    new_runner = Runner(MLP([3, 8, 1]), Square(), Void(), Adam())
    with pytest.raises(ValueError, match="Architecture mismatch"):
        new_runner.warm_start(str(target))
