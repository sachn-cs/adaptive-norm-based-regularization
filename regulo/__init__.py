"""regulo -- Adaptive Norm-Based Regularization for Neural Networks.

Pure-Python (NumPy + SciPy) reproduction of the empirical methodology
from Qasim & Javed (Lund University).  Designed for *transparency* and
*reproducibility* rather than throughput: every gradient flows through
hand-written back-propagation, every regularizer exposes analytical
penalties and gradients, and there are no hidden deep-learning
framework defaults.

Quick start
-----------
>>> from regulo import MLP, Adam, Runner, Ridge, Square
>>> net = MLP([10, 32, 1])
>>> trainer = Runner(net, Square(), Ridge(0.01), Adam(), seed=0)
>>> trainer.fit(X_train, y_train, seed=0)
>>> preds = trainer.predict(X_test)
"""

__version__ = "0.2.0a1"
