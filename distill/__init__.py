"""蒸馏层：teacher / student / 旗舰蒸馏协议 / 基线。"""

from .baselines import train_hard, train_pseudo
from .distiller import DistilFuse
from .students import (
    NumpyLogisticStudent,
    NumpyMLPStudent,
    build_student,
)
from .teachers import AVAILABLE_SKLEARN_TEACHER, SklearnTeacher

__all__ = [
    "SklearnTeacher",
    "AVAILABLE_SKLEARN_TEACHER",
    "NumpyLogisticStudent",
    "NumpyMLPStudent",
    "build_student",
    "DistilFuse",
    "train_hard",
    "train_pseudo",
]
