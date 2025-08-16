import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

import pyter.distributions as pdist

pois = pdist.PoissonSingleHit(jnp.array([0, 0.5, 3, 100, 1]))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_sampling(seed):
    with numpyro.handlers.seed(rng_seed=seed):
        samp1 = numpyro.sample("Test sample", pois)
        samp2 = numpyro.sample("Test sample 2", pois)
    assert all(np.logical_or(samp1 == 1, samp1 == 0))
    assert all(np.logical_or(samp2 == 1, samp2 == 0))
