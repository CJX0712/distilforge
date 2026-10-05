from eval.metrics import accuracy, error_rate, rel_error_drop


def test_accuracy_perfect():
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0


def test_accuracy_half():
    assert abs(accuracy([1, 2], [1, 0]) - 0.5) < 1e-12


def test_error_rate_complement():
    assert error_rate([1, 2], [1, 0]) == 0.5


def test_rel_drop_half():
    assert abs(rel_error_drop(0.2, 0.1) - 0.5) < 1e-12


def test_rel_drop_saturated():
    assert rel_error_drop(0.0, 0.1) == 0.0


def test_rel_drop_negative():
    # 蒸馏比硬标签更差 => 负下降
    assert rel_error_drop(0.1, 0.2) == -1.0
