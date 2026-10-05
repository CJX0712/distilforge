"""合成数据生成（固定 seed，可复现）。

主路径（sklearn 可用）：sklearn.datasets.make_classification
- n_clusters_per_class>1 + hypercube=False + 冗余特征 => 非线性决策边界
- 线性 student（LR）明显弱，非线性 teacher（RF/GBM/MLP）明显强
- 难度旋钮 difficulty -> class_sep 映射（越大越易）

离线兜底（sklearn 缺失）：纯 numpy 多模高斯（保证可跑，质量较低）。
"""

import numpy as np

from core.errors import DataError
from core.types import DatasetSpec

try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def _make_sklearn(spec: DatasetSpec):
    # 甜点设计：信息特征占绝大多数、仅留少量冗余，使线性 student 明显弱于非线性 teacher，
    # 从而保留 KD 暗知识的迁移空间（避免 LR 已近最优导致蒸馏无增益）。
    n_info = max(3, spec.n_features - 2)
    n_redundant = spec.n_features - n_info
    if spec.class_sep is not None:
        sep = float(spec.class_sep)
    else:
        sep = max(0.5, 0.8 + 0.7 * float(spec.difficulty))
    N = spec.n_train + spec.n_test
    X, y = make_classification(
        n_samples=N,
        n_features=spec.n_features,
        n_informative=n_info,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=spec.n_classes,
        n_clusters_per_class=max(1, int(spec.clusters_per_class)),
        class_sep=sep,
        hypercube=False,
        random_state=spec.seed,
        shuffle=True,
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=spec.n_test / N, random_state=spec.seed, stratify=y
    )
    return Xtr, ytr, Xte, yte


def _make_numpy(spec: DatasetSpec):
    rng = np.random.RandomState(spec.seed)
    n_c, n_f = spec.n_classes, spec.n_features
    k = max(int(spec.clusters_per_class), 1)
    n_sub = n_c * k
    sub_centers = rng.randn(n_sub, n_f)
    sub_centers /= np.linalg.norm(sub_centers, axis=1, keepdims=True) + 1e-12
    d = max(float(spec.difficulty), 0.05)
    sep = 0.5 + 0.5 * d
    sub_centers *= sep
    variances = np.ones((n_sub, n_f)) * (1.0 / max(d, 0.15))

    def _sample(n):
        y = rng.randint(0, n_c, size=n)
        X = np.empty((n, n_f), dtype=np.float64)
        for i in range(n):
            c = int(y[i])
            sub = c * k + rng.randint(0, k)
            X[i] = rng.randn(n_f) * np.sqrt(variances[sub]) + sub_centers[sub]
        return X, y

    Xtr, ytr = _sample(spec.n_train)
    rng_test = np.random.RandomState(spec.seed + 777)
    Xte, yte = _sample(rng_test, spec.n_train if False else spec.n_test)
    return Xtr, ytr, Xte, yte


def make_synthetic(spec: DatasetSpec):
    if spec.source != "synthetic":
        raise DataError(f"spec.source={spec.source!r} is not 'synthetic'")
    if spec.n_classes < 2:
        raise DataError("n_classes must be >= 2")
    if spec.n_features < 1:
        raise DataError("n_features must be >= 1")
    if _HAVE_SKLEARN:
        return _make_sklearn(spec)
    return _make_numpy(spec)
