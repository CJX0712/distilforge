"""数据层：合成数据生成 + 数据集载入。"""

from .loaders import AVAILABLE_SKLEARN, load_csv, load_sklearn
from .synthetic import make_synthetic

__all__ = ["make_synthetic", "load_sklearn", "load_csv", "AVAILABLE_SKLEARN"]
