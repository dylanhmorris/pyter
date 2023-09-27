#!/usr/bin/env python3

import pytest

from numpyro.distributions import Normal
from pyter.prior import Prior
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np

def test_prior_bools():
    empty_prior = Prior()
    empty_prior.validate_scheme()
    assert not empty_prior.func_scheme()
    assert not empty_prior.dict_scheme()
    assert empty_prior.no_scheme()
    assert not empty_prior.is_parameterized()
    assert not empty_prior.can_be_parameterized()


def test_parameterize():
    param_prior = Prior(
        parameter="test_parameter",
        distribution=Normal,
        scheme={
            "loc": {
                "value1": 0,
                "value2": 1.25,
                "default": 4},
            "scale": np.array([1.75])
        })

    parameterized = param_prior.parameterize(
        ["value1", "value2", "value3", "value4"])

    assert isinstance(parameterized, Normal)
    assert_array_equal(
        parameterized.loc,
        np.array([0, 1.25, 4, 4]))
    assert_array_equal(
        parameterized.scale,
        np.array([1.75]))
    assert parameterized.batch_shape == (4,)


    fixed_prior = Prior(
        parameter="test_parameter",
        distribution=Normal(0.25, np.array([1,2,3])))

    instantiated_fixed = fixed_prior.parameterize(
        ["value1", "value2", "value3"])

    assert isinstance(instantiated_fixed, Normal)
    assert_array_equal(
        instantiated_fixed.loc,
        np.array([0.25]))
    assert_array_equal(
        instantiated_fixed.scale,
        np.array([1, 2, 3]))
    assert instantiated_fixed.batch_shape == (3,)


def test_no_double_parameterization():
    with pytest.raises(
            ValueError,
            match="Clashing parametrization schemes"):
        should_raise = Prior(
            parameter="test_parameter",
            distribution=Normal(0, 1),
            scheme={"loc": 0, "scale": 1})

def test_scheme_validation():
    with pytest.raises(
            ValueError,
            match=("Parameterization scheme must be "
                   "dictionary, callable, or None")):
        should_raise = Prior(
            parameter="test_parameter",
            distribution=Normal,
            scheme=25)
