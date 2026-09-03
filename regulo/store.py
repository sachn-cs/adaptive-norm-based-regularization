"""Model serialization via NPZ + JSON.

Save and load :class:`regulo.fit.Runner` state to a directory of
NPZ and JSON files.  No ``pickle`` is used; the on-disk format
cannot execute arbitrary code on load.  The schema is stable and
versioned against :data:`regulo.__version__`.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from regulo import __version__
from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import Penalty, REGISTRY

__all__ = ["save", "load", "load_meta", "meta"]

META_FILE = "meta.json"


def meta(runner: Runner) -> Dict:
    """Build the metadata dictionary describing *runner*."""
    from regulo.loss import Softmax

    return {
        "version": __version__,
        "layer_sizes": list(runner.mlp.layer_sizes),
        "loss": runner.loss_fn.name,
        "loss_kwargs": loss_kwargs(runner.loss_fn),
        "penalty": runner.penalty.name,
        "penalty_hp": penalty_hp(runner.penalty),
        "adam": {
            "learning_rate": runner.adam.learning_rate,
            "beta1": runner.adam.beta1,
            "beta2": runner.adam.beta2,
            "epsilon": runner.adam.epsilon,
        },
        "task": runner.task,
    }


def save(runner: Runner, path: str) -> None:
    """Persist a Runner to *path*.

    Creates a directory *path* containing:
      * ``meta.json`` -- architecture and hyperparameters
      * ``weights.npz`` -- one ``w{i}`` array per weight matrix
      * ``biases.npz`` -- one ``b{i}`` array per bias vector
      * ``adam.npz`` -- ``mean`` / ``variance`` / ``clock`` per group
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / META_FILE, "w") as f:
        json.dump(meta(runner), f, indent=2, default=json_default)
    np.savez(
        p / "weights.npz",
        **{f"w{i}": w for i, w in enumerate(runner.mlp.weights)},
    )
    np.savez(
        p / "biases.npz",
        **{f"b{i}": b for i, b in enumerate(runner.mlp.biases)},
    )
    adam_data = {}
    for group in runner.adam.mean:
        for i, buf in enumerate(runner.adam.mean[group]):
            adam_data[f"{group}_{i}_mean"] = (
                np.zeros_like(runner.mlp.weights[i])
                if buf is None
                else buf
            )
        for i, buf in enumerate(runner.adam.variance[group]):
            adam_data[f"{group}_{i}_variance"] = (
                np.zeros_like(runner.mlp.weights[i])
                if buf is None
                else buf
            )
    adam_data["clock"] = np.array(runner.adam.clock)
    np.savez(p / "adam.npz", **adam_data)


def load_meta(path: str) -> Dict:
    """Read and return the metadata dictionary at *path*."""
    p = Path(path)
    with open(p / META_FILE) as f:
        data = json.load(f)
    major = data["version"].split(".")[0]
    lib_major = __version__.split(".")[0]
    if major != lib_major:
        raise ValueError(
            f"Major version mismatch: on-disk {data['version']!r}, "
            f"library {__version__!r}."
        )
    return data


def load(path: str) -> Runner:
    """Reconstruct a Runner from *path*."""
    from regulo.loss import Softmax

    data = load_meta(path)
    mlp = MLP(data["layer_sizes"])
    loss_cls = resolve_loss(data["loss"])
    loss = loss_cls(**data["loss_kwargs"])
    penalty = resolve_penalty(data)
    adam = Adam(**data["adam"])

    p = Path(path)
    weights = np.load(p / "weights.npz")
    biases = np.load(p / "biases.npz")
    for i, w in enumerate(mlp.weights):
        w[...] = weights[f"w{i}"]
    for i, b in enumerate(mlp.biases):
        b[...] = biases[f"b{i}"]

    runner = Runner(mlp, loss, penalty, adam)

    # Restore Adam state if compatible.
    adam_path = p / "adam.npz"
    if adam_path.exists():
        loaded = np.load(adam_path)
        runner.adam.clock = int(loaded["clock"])
        for group in runner.adam.mean:
            for i in range(len(runner.adam.mean[group])):
                key_mean = f"{group}_{i}_mean"
                key_var = f"{group}_{i}_variance"
                if key_mean in loaded.files:
                    runner.adam.mean[group][i] = loaded[key_mean]
                    runner.adam.variance[group][i] = loaded[key_var]
                else:
                    runner.adam.mean[group][i] = np.zeros_like(
                        runner.mlp.weights[i] if group == "weights" else runner.mlp.biases[i]
                    )
                    runner.adam.variance[group][i] = np.zeros_like(
                        runner.mlp.weights[i] if group == "weights" else runner.mlp.biases[i]
                    )
    return runner


def loss_kwargs(loss: Loss) -> Dict:
    return {}


def penalty_hp(penalty: Penalty) -> Dict:
    out = {}
    for name in penalty.hp:
        if name == "c_delta_n":
            continue
        if hasattr(penalty, name):
            out[name] = getattr(penalty, name)
    return out


def resolve_loss(name: str):
    from regulo.loss import Softmax, Square

    table = {"square": Square, "softmax": Softmax}
    if name not in table:
        raise ValueError(f"Unknown loss in meta: {name!r}")
    return table[name]


def resolve_penalty(data: Dict) -> Penalty:
    name = data["penalty"]
    hp = dict(data["penalty_hp"])
    if name in ("covridge", "sparridge"):
        # Re-derive a synthetic C matrix from layer_sizes[0]; the
        # user's actual training data was used to build the original,
        # so the loaded penalty applies only to the first layer.
        p = data["layer_sizes"][0]
        hp["c_delta_n"] = np.eye(p)
    cls = REGISTRY[name]
    return cls(**hp)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
