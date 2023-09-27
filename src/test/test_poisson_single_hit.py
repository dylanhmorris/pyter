#!/usr/bin/env python3

import jax.numpy as jnp
import numpy as np
import numpyro as npro
import numpyro.distributions as ndist
import pyter.distributions as pdist
from pyter.infer import Inference
from pyter.models import AbstractModel
from pyter.data import NullData
from numpy.testing import assert_array_equal, assert_almost_equal

import pytest

pois = pdist.PoissonSingleHit(
    jnp.array([0, 0.5, 3, 100, 1])
)

def test_random_sampling():
    with npro.handlers.seed(rng_seed=5):
        samp1 = npro.sample("Test sample", pois)
        samp2 = npro.sample("Test sample 2", pois)

        comp1 = np.array([0, 0, 0, 1, 1])
        comp2 = np.array([0, 1, 1, 1, 0])

        assert_array_equal(samp1, comp1)
        assert_array_equal(samp2, comp2)

class SimplePSH(AbstractModel):
    def model(self, data=None):
        loglam = npro.sample(
            "log_rate",
            ndist.Normal(0, 1))
        data = npro.sample(
            "observed_hits",
            pdist.PoissonSingleHit(
                jnp.exp(loglam)),
            obs=jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        )
        return loglam

    def validate_data(self, data, run_data):
        pass


# def test_inference_from_distribution():
#     test_inf = Inference()

#     test_inf.infer(
#         model=SimplePSH(),
#         data=NullData(),
#         random_seed=10)

#     assert "log_rate" in test_inf.mcmc_runner.get_samples().keys()
#     assert_almost_equal(
#         np.mean(test_inf.mcmc_runner.get_samples()["log_rate"]),
#         -1.0218989)

#     assert_almost_equal(
#         np.var(test_inf.mcmc_runner.get_samples()["log_rate"]),
#         0.2137335)

