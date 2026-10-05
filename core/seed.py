"""全局确定性种子入口。

唯一 seed 入口：numpy / random / 环境变量级 seed 一次设齐；demo 两次运行
benchmark.json 核心指标必须逐位一致。
"""

import os
import random

import numpy as np

_DEFAULT_SEED = 20261005
_SEEDED = False


def set_all(seed: int = _DEFAULT_SEED) -> None:
    """一次性设齐 numpy / random / PYTHONHASHSEED，尽量覆盖可选 torch。"""
    global _SEEDED
    s = int(seed) & 0xFFFFFFFF
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    try:  # torch 可选，不影响主线
        import torch  # noqa: F401

        torch.manual_seed(s)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass
    _SEEDED = True


def get_seed() -> int:
    return int(os.environ.get("PYTHONHASHSEED", str(_DEFAULT_SEED)))


def is_seeded() -> bool:
    return _SEEDED
