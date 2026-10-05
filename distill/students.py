"""Student 模型：纯 numpy 自实现（确定性、零依赖可跑、支持自定义蒸馏损失）。

两类 student：
- NumpyLogisticStudent：多类逻辑回归（softmax + 交叉熵），主力弱模型
- NumpyMLPStudent：单隐藏层 MLP（tanh），容量介于 LR 与 teacher 之间

蒸馏损失（Hinton 2015）：
    L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1-alpha) * CE(y, softmax(z_s))
对 student logits 的解析梯度（minibatch 平均）：
    dz = [ alpha*T*(q_s - p_t) + (1-alpha)*(p_s - onehot) ] / N
"""

import numpy as np

from core.config import DistilForgeConfig
from core.errors import ModelError
from training.train import minibatches


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    N = y.shape[0]
    oh = np.zeros((N, n_classes), dtype=np.float64)
    oh[np.arange(N), y] = 1.0
    return oh


class NumpyLogisticStudent:
    def __init__(self, n_features: int, n_classes: int):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.W = None
        self.b = None

    def _init_params(self, rng: np.random.RandomState, scale: float = 0.01):
        self.W = rng.randn(self.n_features, self.n_classes) * scale
        self.b = np.zeros(self.n_classes, dtype=np.float64)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.W + self.b

    def _hard_ce(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        p_s = _softmax(self.logits(X))
        return float(-np.mean(np.log(p_s[np.arange(N), y] + 1e-12)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(X), axis=1)

    def _grad(self, X, y, teacher_logits, cfg, alpha):
        N = X.shape[0]
        z = self.logits(X)
        p_s = _softmax(z)
        q_s = _softmax(z / cfg.temperature)
        onehot = _onehot(y, self.n_classes)
        dz = (1.0 - alpha) * (p_s - onehot) / N
        if alpha > 0.0 and teacher_logits is not None:
            p_t = _softmax(teacher_logits / cfg.temperature)
            dz = dz + alpha * cfg.temperature * (q_s - p_t) / N
        dW = X.T @ dz + cfg.weight_decay * self.W
        db = dz.sum(axis=0)
        return dW, db

    def _train(self, X, y, teacher_logits, cfg, rng, alpha):
        if self.W is None:
            self._init_params(rng)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        for _epoch in range(cfg.n_epochs):
            for mb in minibatches(rng, X.shape[0], cfg.batch_size):
                dW, db = self._grad(
                    X[mb],
                    y[mb],
                    teacher_logits[mb] if teacher_logits is not None else None,
                    cfg,
                    alpha,
                )
                self.W -= cfg.lr * dW
                self.b -= cfg.lr * db
        return self

    def fit_distill(self, X, teacher_logits, y, cfg, rng):
        return self._train(X, y, teacher_logits, cfg, rng, cfg.alpha)

    def fit_hard(self, X, y, cfg, rng):
        return self._train(X, y, None, cfg, rng, 0.0)


class NumpyMLPStudent:
    def __init__(self, n_features: int, n_classes: int, n_hidden: int = 32):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_hidden = int(n_hidden)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _init_params(self, rng: np.random.RandomState, scale: float = 0.01):
        self.W1 = rng.randn(self.n_features, self.n_hidden) * scale
        self.b1 = np.zeros(self.n_hidden, dtype=np.float64)
        self.W2 = rng.randn(self.n_hidden, self.n_classes) * scale
        self.b2 = np.zeros(self.n_classes, dtype=np.float64)

    def _forward(self, X):
        a1 = X @ self.W1 + self.b1
        h = np.tanh(a1)
        z = h @ self.W2 + self.b2
        return h, z

    def logits(self, X: np.ndarray) -> np.ndarray:
        _h, z = self._forward(np.asarray(X, dtype=np.float64))
        return z

    def _hard_ce(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        p_s = _softmax(self.logits(X))
        return float(-np.mean(np.log(p_s[np.arange(N), y] + 1e-12)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(X), axis=1)

    def _grad(self, X, y, teacher_logits, cfg, alpha):
        N = X.shape[0]
        h, z = self._forward(X)
        p_s = _softmax(z)
        q_s = _softmax(z / cfg.temperature)
        onehot = _onehot(y, self.n_classes)
        dz = (1.0 - alpha) * (p_s - onehot) / N
        if alpha > 0.0 and teacher_logits is not None:
            p_t = _softmax(teacher_logits / cfg.temperature)
            dz = dz + alpha * cfg.temperature * (q_s - p_t) / N
        dW2 = h.T @ dz + cfg.weight_decay * self.W2
        db2 = dz.sum(axis=0)
        dh = (dz @ self.W2.T) * (1.0 - h**2)
        dW1 = X.T @ dh + cfg.weight_decay * self.W1
        db1 = dh.sum(axis=0)
        return dW1, db1, dW2, db2

    def _train(self, X, y, teacher_logits, cfg, rng, alpha):
        if self.W1 is None:
            self._init_params(rng)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        for _epoch in range(cfg.n_epochs):
            for mb in minibatches(rng, X.shape[0], cfg.batch_size):
                tt = teacher_logits[mb] if teacher_logits is not None else None
                dW1, db1, dW2, db2 = self._grad(X[mb], y[mb], tt, cfg, alpha)
                self.W1 -= cfg.lr * dW1
                self.b1 -= cfg.lr * db1
                self.W2 -= cfg.lr * dW2
                self.b2 -= cfg.lr * db2
        return self

    def fit_distill(self, X, teacher_logits, y, cfg, rng):
        return self._train(X, y, teacher_logits, cfg, rng, cfg.alpha)

    def fit_hard(self, X, y, cfg, rng):
        return self._train(X, y, None, cfg, rng, 0.0)


def build_student(n_features: int, n_classes: int, cfg: DistilForgeConfig):
    if cfg.student_type == "logistic":
        return NumpyLogisticStudent(n_features, n_classes)
    if cfg.student_type == "mlp":
        return NumpyMLPStudent(n_features, n_classes, n_hidden=cfg.n_hidden)
    raise ModelError(f"不支持的 student_type: {cfg.student_type}")
