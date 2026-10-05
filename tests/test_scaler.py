import numpy as np

from preprocess.scaler import StandardScaler


def test_fit_transform_zero_mean_unit_std():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    s = StandardScaler().fit(X)
    t = s.transform(X)
    assert abs(float(t.mean())) < 1e-9
    assert abs(float(t.std()) - 1.0) < 1e-6


def test_transform_independent_of_test():
    # 关键防泄漏：transform 必须仅使用训练集统计量（mean/scale），与测试集无关
    Xtr = np.array([[1.0], [2.0], [3.0]])
    Xte = np.array([[10.0], [20.0]])
    s = StandardScaler().fit(Xtr)
    t = s.transform(Xte)
    mean_tr = Xtr.mean(axis=0)
    std_tr = Xtr.std(axis=0) + 1e-9
    expected = (Xte - mean_tr) / std_tr
    assert np.allclose(t, expected, atol=1e-9)
