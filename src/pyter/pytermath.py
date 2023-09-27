#!/usr/bin/env python3

"""
pytermath defines needed
mathematical functions for
the PyTer package, aimed
at numerical stability
and autodiff compability
"""

import jax.numpy as jnp
from jax import custom_jvp


@custom_jvp
def log1m_exp(x):
    """Numerically stable calculation
    of the quantity log(1 - exp(x)),
    following the algorithm of
    Machler [1]. This is
    the algorithm used in TensorFlow Probability,
    PyMC, and Stan, but it is not provided
    yet with Numpyro.

    Currently returns NaN for x > 0,
    but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/
    Rmpfr/vignettes/log1mexp-note.pdf

    Parameters
    ----------
    x :


    Returns
    -------

    """
    # return 0. rather than -0. if
    # we get a negative exponent that exceeds
    # the floating point representation
    arr_x = 1.0 * jnp.array(x)
    oob = arr_x < jnp.log(jnp.finfo(
        arr_x.dtype).smallest_normal)
    mask = arr_x > -0.6931472  # appox -log(2)
    more_val = jnp.log(-jnp.expm1(arr_x))
    less_val = jnp.log1p(-jnp.exp(arr_x))

    return jnp.where(
        oob,
        0.,
        jnp.where(
            mask,
            more_val,
            less_val))


@log1m_exp.defjvp
def log1m_exp_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    ans = log1m_exp(x)
    ans_dot = x_dot * -1. * jnp.exp(
        x - ans)
    return ans, ans_dot


def log1p_exp_scalar(x):
    """Valuable for checking the below
    vectorized jax.lax implementation
    of the algorithm of Machler [1]

    [1] https://cran.r-project.org/web/packages/
    Rmpfr/vignettes/log1mexp-note.pdf

    Parameters
    ----------
    x :


    Returns
    -------

    """
    result = None
    if x <= -37:
        result = jnp.exp(x)
    elif x <= 18:
        result = jnp.log1p(jnp.exp(x))
    elif x <= 33.3:
        result = x + jnp.exp(-x)
    else:
        result = x
    return result


def log1p_exp(x):
    """Stably calculate log(1 + exp(x))
    according to the
    algorithm of Machler [1]

    [1] https://cran.r-project.org/web/packages/Rmpfr/
    vignettes/log1mexp-note.pdf

    Parameters
    ----------
    x :


    Returns
    -------

    """
    return jnp.where(
        x <= 18.0,
        jnp.log1p(jnp.exp(x)),
        x + jnp.exp(-x))


def log_diff_exp(a, b):
    """

    Parameters
    ----------
    a :
    b :

    Returns
    -------

    """
    # note that following Stan,
    # we want the log diff exp
    # of -inf, -inf to be -inf,
    # not nan, because that
    # corresponds to log(0 - 0) = -inf
    return ((1. * a) + log1m_exp(
        1. * b - 1. * a))
