import pytest
from pyter.pytermath import log_diff_exp
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal
import numpy as np


def test_log_diff_exp():
    """
    Test that log_diff_exp
    function is numerically
    stable but also agrees
    with the naive implementation
    where it that function is stable
    """

    def naive(a, b):
        return jnp.log(jnp.exp(a) - jnp.exp(b))

    np.random.seed(523)
    small_to_medium_a = 10 * (np.random.random(100) - 1)
    small_to_medium_b = 10 * (np.random.random(100) - 1)

    assert_array_almost_equal(
        naive(small_to_medium_a, small_to_medium_b),
        log_diff_exp(small_to_medium_a,
                     small_to_medium_b))
