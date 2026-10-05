"""训练层：通用 SGD 工具。"""

from .train import constant_lr, cosine_lr, minibatches

__all__ = ["minibatches", "cosine_lr", "constant_lr"]
