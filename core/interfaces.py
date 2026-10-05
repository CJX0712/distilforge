"""接口协议（单向无环，便于替换与测试）。"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
