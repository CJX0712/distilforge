"""Teacher 封装：Tier-0 顶级开源强模型（scikit-learn RF/GBM/MLP/Logistic）。

统一暴露 logits(X) = log(predict_proba(X) + eps)：
- 温度 T=1 时 softmax(logits) 退化为 teacher 真实概率（标准 KD 直觉）
- 温度 T>1 时对概率分布软化，暴露类间结构信息
"""

import numpy as np

from core.errors import DependencyError, ModelError

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    AVAILABLE_SKLEARN_TEACHER = True
except Exception:
    AVAILABLE_SKLEARN_TEACHER = False


class SklearnTeacher:
    def __init__(self, teacher_type: str = "random_forest", seed: int = 20261005):
        if not AVAILABLE_SKLEARN_TEACHER:
            raise DependencyError("scikit-learn 不可用，无法构建 SklearnTeacher")
        self.teacher_type = teacher_type
        self.seed = int(seed)
        self.model = self._build()
        self.n_classes_ = None

    def _build(self):
        t = self.teacher_type
        if t == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=1)
        if t == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.seed)
        if t == "mlp":
            return MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=600, random_state=self.seed)
        if t == "logistic":
            return LogisticRegression(max_iter=3000, random_state=self.seed)
        raise ModelError(f"不支持的 teacher_type: {t}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnTeacher":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.model.fit(X, y)
        self.n_classes_ = int(getattr(self.model, "n_classes_", len(np.unique(y))))
        return self

    def probs(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def logits(self, X: np.ndarray) -> np.ndarray:
        """log(predict_proba + eps)，作为温度软化的 logit 尺度输入。"""
        p = self.probs(X) + 1e-9
        return np.log(p)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=int)
