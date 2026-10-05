"""通用 SGD 工具（确定性，依赖注入的 numpy RandomState）。"""

import math

import numpy as np


def minibatches(rng: np.random.RandomState, n: int, batch_size: int):
    idx = rng.permutation(n)
    bs = max(1, int(batch_size))
    for s in range(0, n, bs):
        yield idx[s : s + bs]


def constant_lr(base: float, epoch: int, total: int) -> float:
    return base


def cosine_lr(base: float, epoch: int, total: int) -> float:
    total = max(total, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * epoch / total))
