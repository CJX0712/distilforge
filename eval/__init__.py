"""评估层：统一指标口径（accuracy 越大越好）。"""

from .metrics import accuracy, error_rate, rel_error_drop

__all__ = ["accuracy", "error_rate", "rel_error_drop"]
