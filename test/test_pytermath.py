import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal

from pyter.pytermath import log1m_exp, log_diff_exp


def test_log1m_exp():
    """
    Test that the log1m_exp implementation
    agrees with the naive implementation where
    that is stable
    """

    def naive(a):
        return jnp.log(1 - jnp.exp(a))

    np.random.seed(52352)
    small_to_medium_a = 10 * (np.random.random(100) - 1)

    assert_array_almost_equal(
        naive(small_to_medium_a),
        log1m_exp(small_to_medium_a),
    )


def test_log_diff_exp():
    """
    Test that log_diff_exp agrees
    with the naive implementation
    where that function is stable
    """

    def naive(a, b):
        return jnp.log(jnp.exp(a) - jnp.exp(b))

    np.random.seed(532)
    small_to_medium_a = 10 * (np.random.random(100) - 1)
    small_to_medium_b = 10 * (np.random.random(100) - 1)

    assert_array_almost_equal(
        naive(small_to_medium_a, small_to_medium_b),
        log_diff_exp(small_to_medium_a, small_to_medium_b),
    )
