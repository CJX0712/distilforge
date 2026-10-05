"""流水线层：编排 teacher/student/baseline 训练与多 seed 评测。"""

from .distill_pipeline import DistillPipeline

__all__ = ["DistillPipeline"]
