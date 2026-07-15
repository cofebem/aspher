import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import aspher as hc


def test_models():
    p = np.array([1.0, 0.5, 0.0, -0.1, 2.0, 0.0])
    v = np.array([0.0, 1.0, 2.0, 3.0, 0.5, 0.0])
    T = np.full(6, 300.0)

    tr = hc.TrescaFriction(0.07)
    s = np.asarray(tr.threshold(p, v, T))
    assert s.shape == (6,)
    assert np.allclose(s[[0, 1, 4]], 0.07)
    assert np.all(s[[2, 3, 5]] == 0.0)      # p <= 0 -> s = 0
    assert not tr.velocity_dependent

    co = hc.CoulombFriction(0.3)
    s = np.asarray(co.threshold(p, v, T))
    assert np.isclose(s[0], 0.3) and np.isclose(s[1], 0.15) and s[3] == 0.0
    assert not co.velocity_dependent

    # user law: rate-weakening; sanitizer must clamp the p<0 negative to 0
    mu0, v0 = 0.3, 1.0
    user = hc.UserFriction(lambda p, v, T: mu0 * p / (1.0 + v / v0),
                           velocity_dependent=True)
    s = np.asarray(user.threshold(p, v, T))
    assert np.isclose(s[0], 0.3) and np.isclose(s[1], 0.15 / 2.0)
    assert s[3] == 0.0 and s[2] == 0.0
    assert user.velocity_dependent

    # a NaN-producing callback is sanitized to 0 at that point
    bad = hc.UserFriction(lambda p, v, T: np.where(p > 1.5, np.nan, mu0 * p),
                          velocity_dependent=False)
    s = np.asarray(bad.threshold(p, v, T))
    assert s[4] == 0.0 and np.isclose(s[0], 0.3)
    print("test_models: OK")


if __name__ == "__main__":
    test_models()
    print("test_friction_py: all checks passed")
