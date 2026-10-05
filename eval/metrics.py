"""统一评估指标。"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - accuracy(y_true, y_pred)


def rel_error_drop(err_hard: float, err_distill: float) -> float:
    """相对错误率下降：(err_hard - err_distill) / err_hard。

    >0 表示蒸馏优于硬标签基线；err_hard<=0 视为饱和（返回 0）。
    """
    if err_hard <= 0:
        return 0.0
    return (err_hard - err_distill) / err_hard
