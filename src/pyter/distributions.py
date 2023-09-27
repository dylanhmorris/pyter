#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Custom probability distributions for quantitative virology.

Distributions are subclasses of
:class:`numpyro.distributions.Distribution
<numpyro.distributions.distribution.Distribution>`,
which we will refer to simply as class
:class:`~numpyro.distributions.distribution.Distribution`.
"""

import numpyro as npro
import jax.numpy as jnp
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.util import (
    validate_sample,
    promote_shapes
)
from numpyro.util import is_prng_key

import pyter.constraints as pyterconstraints


class PoissonSingleHit(npro.distributions.Distribution):
    """
    Poisson Single-Hit Distribution

    This is a distribution that yields a 1
    if a Poisson random variable is non-zero
    and a zero otherwise. It occurs in virology
    because if we expose a set of cells to some
    quantity of infectious virus particles
    ("virions"), the number that succesfully enter
    a cell and replicate can be modeled as a
    Poisson distributed random variable with a mean
    related to the initial quantity of virions.
    The probability of seeing any evidence of cell
    invasion is then equal to the probability that
    this Poisson random variable is non-zero
    (i.e. at least one virion successfully
    invaded a cell).

    Parameters
    ----------
    rate : :py:class:`float`
        The rate of the Poisson random variable.
    """

    arg_constraints = {
        "rate": pyterconstraints.nonnegative}

    support = constraints.boolean

    def __init__(self,
                 rate=0,
                 validate_args=None):

        self.rate = rate
        batch_shape = jnp.shape(self.rate)

        self.bernoulli_ = npro.distributions.Bernoulli(
            probs=1 - jnp.exp(-self.rate),
            validate_args=True)

        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """
        Parameters
        ----------
        key :
        sample_shape :
             (Default value = ())

        Returns
        -------

        """
        assert is_prng_key(key)
        return self.bernoulli_.sample(
            key,
            sample_shape=sample_shape)

    @validate_sample
    def log_prob(self, value):
        """

        Parameters
        ----------
        value :

        Returns
        -------

        """
        return self.bernoulli_.log_prob(value)


class TiterPlate(npro.distributions.Distribution):
    """
    Base distribution to represent a set of titers

    Subclasses represent different assays:
    :class:`PlaquePlate` for plaque assays,
    and :class:`EndpointTiterPlate` for
    endpoint titration assays.
    """
    arg_constraints = {
        "log_titer": constraints.real,
        "log_dilution": constraints.real,
        "log_base": constraints.positive,
        "well_volume": constraints.positive,
        "false_hit_rate": pyterconstraints.nonnegative
    }

    def __init__(self,
                 log_titer=None,
                 log_dilution=None,
                 log_base=10,
                 well_volume=1,
                 false_hit_rate=0,
                 validate_args=None):
        (self.log_titer,
         self.log_dilution,
         self.log_base,
         self.well_volume,
         self.false_hit_rate) = promote_shapes(
             log_titer,
             log_dilution,
             log_base,
             well_volume,
             false_hit_rate)

        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.log_titer),
            jnp.shape(self.log_dilution),
            jnp.shape(self.log_base),
            jnp.shape(self.well_volume),
            jnp.shape(self.false_hit_rate))

        super().__init__(validate_args=validate_args,
                         batch_shape=batch_shape)


class PlaquePlate(TiterPlate):
    """
    Distribution class to represent
    a set of titers quantified
    by a plaque assay.
    """

    support = constraints.nonnegative_integer

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.hit_rate = (
            self.false_hit_rate +
            self.well_volume * jnp.exp(
                jnp.log(self.log_base) *
                (self.log_titer + self.log_dilution))
        )

        self.poisson_ = npro.distributions.Poisson(
            rate=self.hit_rate)

    def sample(self, key, sample_shape=()):
        """

        Parameters
        ----------
        key :

        sample_shape :
             (Default value = ())

        Returns
        -------

        """
        assert is_prng_key(key)
        return self.poisson_.sample(
            key,
            sample_shape=sample_shape)

    @validate_sample
    def log_prob(self, value):
        """

        Parameters
        ----------
        value :

        Returns
        -------

        """
        return self.poisson_.log_prob(value)


class EndpointTiterPlate(TiterPlate):
    """
    Distribution class to represent
    a set of titers quantified
    by endpoint titration.
    """

    support = constraints.boolean

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.single_hit_rate = (
            self.false_hit_rate +
            self.well_volume *
            jnp.log(2) *  # convert id50 to hit units
            jnp.exp(
                jnp.log(self.log_base) *
                (self.log_titer + self.log_dilution))
        )

        self.single_hit_ = PoissonSingleHit(
            rate=self.single_hit_rate)

    def sample(self, key, sample_shape=()):
        """

        Parameters
        ----------
        key :

        sample_shape :
             (Default value = ())

        Returns
        -------

        """
        assert is_prng_key(key)
        return self.single_hit_.sample(
            key,
            sample_shape=sample_shape)

    @validate_sample
    def log_prob(self, value):
        """

        Parameters
        ----------
        value :

        Returns
        -------

        """
        return self.single_hit_.log_prob(value)
