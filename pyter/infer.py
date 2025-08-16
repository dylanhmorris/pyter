#!/usr/bin/env python3

# filename: infer.py
# description: main Inference
# class that will handle all our
# inferential tasks in PyTer

import attrs
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

from pyter.data import AbstractData
from pyter.models import AbstractModel


@attrs.define
class Inference:
    """ """

    target_accept_prob: float = 0.8
    max_tree_depth: int = 11
    forward_mode_differentiation: bool = False
    mcmc_runner: MCMC = None
    kernel: NUTS = None
    run_model: AbstractModel = None
    _run_reparam_model: object = None
    run_data: dict = None
    run_rng_key: jax.random.PRNGKey = None

    def new_kernel(self, model):
        """

        Parameters
        ----------
        model :

        Returns
        -------

        """
        return NUTS(
            model,
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
            forward_mode_differentiation=self.forward_mode_differentiation,
        )

    def new_runner(self, kernel, num_warmup, num_samples, **kwargs):
        """

        Parameters
        ----------
        kernel :
        num_warmup :
        num_samples :
        **kwargs :

        Returns
        -------

        """
        return MCMC(
            kernel, num_warmup=num_warmup, num_samples=num_samples, **kwargs
        )

    def infer(
        self,
        model: AbstractModel = None,
        data: AbstractData = None,
        random_seed: int = None,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        validate_data: bool = True,
        **kwargs,
    ):
        """Conduct inference.

        Draw posterior samples from the
        given model with the given
        data

        Parameters
        ----------
        model: AbstractModel :
             (Default value = None)
        data: AbstractData :
             (Default value = None)
        random_seed: int :
             (Default value = None)
        num_warmup: int :
             (Default value = 1000)
        num_samples: int :
             (Default value = 1000)
        validate_data: bool :
             (Default value = True)
        **kwargs :


        Returns
        -------

        """
        self.run_model = model
        self.kernel = self.new_kernel(self.run_model.get_reparam())
        self.mcmc_runner = self.new_runner(
            self.kernel, num_warmup, num_samples, **kwargs
        )

        if random_seed is None:
            random_seed = np.random.randint(0, 100000)

        # this saves state so that we know
        # exactly what we used for the run
        self.run_rng_key = jax.random.PRNGKey(random_seed)
        self.run_data = data.freeze()

        if validate_data:
            self.run_model.validate_data(data, self.run_data)

        self.mcmc_runner.run(self.run_rng_key, data=self.run_data)
