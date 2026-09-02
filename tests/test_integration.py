"""Integration tests: end-to-end training with all regularizers."""

import numpy as np
from sklearn.preprocessing import StandardScaler

import regulo.losses as losses
import regulo.metrics as metrics
from regulo.cv import grid_search_cv
from regulo.network import FullyConnectedNetwork
from regulo.optimizer import Adam
from regulo.regularizers import (
    Covridge,
    ElasticNet,
    Lasso,
    NoRegularizer,
    Ridge,
    Sparridge,
)
from regulo.trainer import Trainer


def test_train_regression_decreases_loss():
    np.random.seed(0)
    x = np.random.randn(64, 5)
    y = (
        x @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]).reshape(-1, 1)
        + np.random.randn(64, 1) * 0.1
    )
    net = FullyConnectedNetwork([5, 8, 1])
    opt = Adam(learning_rate=1e-2)
    reg = Ridge(lambda_=1e-4)
    trainer = Trainer(
        net, losses.MSELoss(), reg, opt, batch_size=16, epochs=100
    )
    trainer.fit(x, y)
    final_loss = trainer.history["train_loss"][-1]
    initial_loss = trainer.history["train_loss"][0]
    assert final_loss < initial_loss
    preds = trainer.predict(x)
    assert preds.shape == y.shape


def test_classification_forward_shape():
    np.random.seed(0)
    x = np.random.randn(32, 4)
    y = np.random.randint(0, 3, size=(32,))
    net = FullyConnectedNetwork([4, 8, 3])
    opt = Adam(learning_rate=1e-2)
    reg = Ridge(lambda_=1e-4)
    trainer = Trainer(
        net, losses.CrossEntropyLoss(), reg, opt, batch_size=16, epochs=50
    )
    trainer.fit(x, y)
    logits = trainer.predict(x)
    assert logits.shape == (32, 3)
    preds = np.argmax(logits, axis=1)
    acc = metrics.balanced_accuracy_score(y, preds)
    assert 0.0 <= acc <= 1.0


def test_cv_grid_search_runs():
    np.random.seed(0)
    x = np.random.randn(60, 4)
    y = (
        x @ np.array([1.0, -1.0, 0.5, 0.0]).reshape(-1, 1)
        + np.random.randn(60, 1) * 0.1
    )
    param_grid = [{"lambda_": 1e-3}, {"lambda_": 1e-2}]
    best_params, best_score = grid_search_cv(
        x,
        y,
        layer_sizes=[4, 4, 1],
        method="ridge",
        param_grid=param_grid,
        loss_fn=losses.MSELoss(),
        n_splits=3,
        epochs=20,
    )
    assert "lambda_" in best_params
    assert best_score <= 0  # negative MSE


def test_train_with_each_regularizer():
    np.random.seed(0)
    x = np.random.randn(40, 4)
    y = np.random.randn(40, 1)
    c = np.eye(4)

    regularizers = [
        NoRegularizer(),
        Ridge(lambda_=1e-4),
        Lasso(gamma=1e-4),
        ElasticNet(alpha=0.5, gamma=1e-4),
        Covridge(lambda1=1e-4, lambda2=1e-4, c_delta_n=c),
        Sparridge(lambda1=1e-4, gamma=1e-4, c_delta_n=c),
    ]

    for reg in regularizers:
        net = FullyConnectedNetwork([4, 4, 1])
        opt = Adam(learning_rate=1e-2)
        trainer = Trainer(
            net, losses.MSELoss(), reg, opt, batch_size=16, epochs=20
        )
        trainer.fit(x, y)
        assert len(trainer.history["train_loss"]) == 20


def test_early_stopping():
    np.random.seed(0)
    x = np.random.randn(64, 4)
    y = np.random.randn(64, 1)
    net = FullyConnectedNetwork([4, 4, 1])
    opt = Adam(learning_rate=1e-2)
    reg = Ridge(lambda_=1e-4)
    trainer = Trainer(
        net,
        losses.MSELoss(),
        reg,
        opt,
        batch_size=16,
        epochs=500,
        early_stopping=True,
        patience=5,
    )
    # Provide validation data to trigger early stopping.
    trainer.fit(x[:40], y[:40], x[40:], y[40:])
    assert len(trainer.history["train_loss"]) < 500


def test_feature_standardization_consistency():
    np.random.seed(0)
    x = np.random.randn(100, 4)
    y = np.random.randn(100, 1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[:80])
    x_test = scaler.transform(x[80:])
    net = FullyConnectedNetwork([4, 4, 1])
    opt = Adam(learning_rate=1e-2)
    reg = Ridge(lambda_=1e-4)
    trainer = Trainer(net, losses.MSELoss(), reg, opt, batch_size=16, epochs=10)
    trainer.fit(x_train, y[:80])
    preds = trainer.predict(x_test)
    assert preds.shape == (20, 1)


def test_trainer_history_keys():
    np.random.seed(0)
    x = np.random.randn(32, 3)
    y = np.random.randn(32, 1)
    net = FullyConnectedNetwork([3, 2, 1])
    opt = Adam()
    reg = Ridge(lambda_=1e-4)
    trainer = Trainer(net, losses.MSELoss(), reg, opt, epochs=5)
    trainer.fit(x, y)
    assert "train_loss" in trainer.history
    assert len(trainer.history["train_loss"]) == 5
    assert "val_loss" in trainer.history
    assert len(trainer.history["val_loss"]) == 0


def test_trainer_with_validation_history():
    np.random.seed(0)
    x = np.random.randn(32, 3)
    y = np.random.randn(32, 1)
    net = FullyConnectedNetwork([3, 2, 1])
    opt = Adam()
    reg = Ridge(lambda_=1e-4)
    trainer = Trainer(net, losses.MSELoss(), reg, opt, epochs=5)
    trainer.fit(x[:20], y[:20], x[20:], y[20:])
    assert len(trainer.history["val_loss"]) == 5
