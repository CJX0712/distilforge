import numpy as np

from core.seed import get_seed, set_all


def test_set_all_seed_value():
    set_all(42)
    assert get_seed() == 42
    set_all(7)
    assert get_seed() == 7


def test_set_all_reproducible_rng():
    set_all(123)
    a = np.random.RandomState(123).randn(5)
    set_all(123)
    b = np.random.RandomState(123).randn(5)
    assert np.array_equal(a, b)
