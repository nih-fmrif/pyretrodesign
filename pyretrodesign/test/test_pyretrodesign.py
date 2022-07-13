from scipy import stats
import numpy as np
from ..pyretrodesign import pyretrodesign

# print(np.random.SeedSequence().entropy)
# produced 217392773431744244676552396159683491339

def test_stats():
    A = 0.5
    s = 1
    alpha = 0.05
    df = np.inf
    z = stats.t.ppf(1 - (alpha / 2), df)
    p_hi = 1 - stats.t.cdf(z - (A / s), df)
    p_lo = stats.t.cdf(-z - (A / s), df)
    assert np.isclose(z, 1.959963984540054)
    assert np.isclose(p_hi, 0.07214998621588009)
    assert np.isclose(p_lo,  0.00694754794471637)


def test_pyretrodesign():
    A = 2.8
    s = 1

    power, type_s, exaggeration, est_df = pyretrodesign(A, s)
    assert power == 0.7995568714356515
    assert type_s == 1.2108426227178002e-06
    assert np.isclose(exaggeration, 1.1284062480156192, rtol=1e-02, atol=1e-02)