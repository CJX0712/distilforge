"""数据集载入：sklearn 内置数据集 + 本地 CSV。"""

import csv

import numpy as np

from core.errors import DataError, DependencyError

try:  # sklearn 为 Tier-0 可选后端，缺失时仅影响内置数据集载入
    from sklearn import datasets as _skdatasets

    AVAILABLE_SKLEARN = True
except Exception:
    _skdatasets = None
    AVAILABLE_SKLEARN = False


def load_sklearn(name: str, seed: int = 20261005):
    if not AVAILABLE_SKLEARN:
        raise DependencyError("scikit-learn 不可用，无法载入内置数据集")
    name = name.lower()
    if name == "digits":
        d = _skdatasets.load_digits()
    elif name == "iris":
        d = _skdatasets.load_iris()
    elif name == "wine":
        d = _skdatasets.load_wine()
    else:
        raise DataError(f"未知 sklearn 数据集: {name!r}")
    X = np.asarray(d.data, dtype=np.float64)
    y = np.asarray(d.target, dtype=int)
    return X, y


def load_csv(path: str, label_col: int = -1):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise DataError(f"CSV 解析失败或为空: {path}")
    if label_col < 0:
        label_col += arr.shape[1]
    y = arr[:, label_col].astype(int)
    X = np.delete(arr, label_col, axis=1)
    return X, y
