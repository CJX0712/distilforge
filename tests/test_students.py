import numpy as np

from core.config import DistilForgeConfig
from core.types import DatasetSpec
from data.synthetic import make_synthetic
from distill.baselines import train_hard
from distill.students import NumpyLogisticStudent, NumpyMLPStudent, build_student


def _spec(difficulty=1.2):
    # 二分类、单子高斯、大分离度 → 线性可分，作为"模型能学习"的健全性基线
    # （多类合成数据含非线性边界，线性/逻辑模型未必能学；基准改用 MLP student）
    return DatasetSpec(
        name="t",
        n_classes=2,
        n_features=8,
        n_train=1000,
        n_test=400,
        difficulty=difficulty,
        clusters_per_class=1,
        class_sep=5.0,
        seed=1,
        source="synthetic",
    )


def test_logistic_beats_random():
    spec = _spec()
    Xtr, ytr, Xte, yte = make_synthetic(spec)
    cfg = DistilForgeConfig(student_type="logistic", n_epochs=300)
    s = NumpyLogisticStudent(8, 2)
    train_hard(s, Xtr, ytr, cfg, np.random.RandomState(0))
    assert (s.predict(Xte) == yte).mean() > 0.5


def test_mlp_beats_random():
    spec = _spec()
    Xtr, ytr, Xte, yte = make_synthetic(spec)
    cfg = DistilForgeConfig(student_type="mlp", n_hidden=16, n_epochs=300)
    s = NumpyMLPStudent(8, 2, n_hidden=16)
    train_hard(s, Xtr, ytr, cfg, np.random.RandomState(0))
    assert (s.predict(Xte) == yte).mean() > 0.5


def test_build_student_factory():
    cfg = DistilForgeConfig(student_type="mlp", n_hidden=8)
    s = build_student(5, 3, cfg)
    assert isinstance(s, NumpyMLPStudent)
    assert s.n_hidden == 8


def test_predict_proba_sums_to_one():
    spec = _spec()
    Xtr, ytr, Xte, _ = make_synthetic(spec)
    cfg = DistilForgeConfig()
    s = NumpyLogisticStudent(8, 2)
    train_hard(s, Xtr, ytr, cfg, np.random.RandomState(0))
    p = s.predict_proba(Xte)
    assert p.shape[1] == 2
    assert np.allclose(p.sum(axis=1), 1.0)


def test_determinism_same_seed():
    spec = _spec()
    Xtr, ytr, Xte, _ = make_synthetic(spec)
    cfg = DistilForgeConfig()
    s1 = NumpyLogisticStudent(8, 2)
    train_hard(s1, Xtr, ytr, cfg, np.random.RandomState(0))
    s2 = NumpyLogisticStudent(8, 2)
    train_hard(s2, Xtr, ytr, cfg, np.random.RandomState(0))
    assert np.array_equal(s1.predict(Xte), s2.predict(Xte))


def test_grad_vs_numeric():
    # 解析梯度与数值梯度一致性（核心数学正确性不变量）
    spec = _spec()
    Xtr, ytr, _, _ = make_synthetic(spec)
    cfg = DistilForgeConfig(alpha=0.0)  # 仅硬 CE
    s = NumpyLogisticStudent(8, 2)
    s._init_params(np.random.RandomState(0))
    Xb, yb = Xtr[:8], ytr[:8]
    dW, db = s._grad(Xb, yb, None, cfg, 0.0)

    eps = 1e-6
    W0 = s.W.copy()
    s.W = W0.copy()
    # 数值梯度对某个权重
    num = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            s.W = W0.copy()
            s.W[i, j] += eps
            lp = s._hard_ce(Xb, yb)
            s.W = W0.copy()
            s.W[i, j] -= eps
            lm = s._hard_ce(Xb, yb)
            num[i, j] = (lp - lm) / (2 * eps)
    assert np.allclose(dW, num, atol=1e-3)
