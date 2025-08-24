"""
The :mod:`~pyter.models` module provides
flexible model classes to represent
distinct experimental setups.

The :class:`AbstractModel` base class
serves as a template for :class:`Model`
subclasses that represent different
possible experimental setups with different
inferred quantities of interest.

The core of any :class:`Model` subclass is
the :meth:`~pyter.models.AbstractModel.model`
method, which takes in a data dictionary
and makes calls to :func:`numpyro.sample()
<numpyro.primitives.sample>` to
define the stochastic generative process.

"""

import inspect

import attrs
import jax
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpy.typing import ArrayLike
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

import pyter.data as pdata
from pyter.distributions import EndpointTiterPlate, PlaquePlate, TiterPlate


def well_distribution_factory(
    assay: str,
    log_titer: ArrayLike,
    log_dilution: ArrayLike,
    log_base: ArrayLike,
    well_volume: ArrayLike,
    false_hit_rate: ArrayLike,
    validate_args: bool = True,
) -> TiterPlate:
    """
    Get an appropriate distribution for titer wells.

    Each entry of the various array inputs represents
    exactly one titration well.

    Parameters
    ----------
    assay : :class:`str` = {'pfu', 'tcid'}
        Which titration assay to use. Options are
        ``'pfu'``--plaque assay--and
        ``'tcid'``--endpoint titration assay.

    log_titer : :data:`~numpy.typing.ArrayLike`
        Underlying log titer(s) per unit volume in
        the undilute sample(s).

    log_dilution : :data:`~numpy.typing.ArrayLike`
        Log dilution(s) relative to the original
        sample(s) for each well's inoculum.

    log_base ~numpy.typing.ArrayLike`
        Base of the logarithim for logarithmic
        quantities including titer and
        dilution (e.g. e, 2, 10, etc).

    well_volume : :data:`~numpy.typing.ArrayLike`
        Volume of the inoculum delivered to
        each well, in the same units as the
        per unit volume for the ``log_titer``
        values. So if log titers are given per
        mL, this is the volume of inoculum
        in mL.

    false_hit_rate : :data:`~numpy.typing.ArrayLike`
        Rate (mean number per well)
        of false hits (i.e. rate of apparent
        infection with a sample containing
        no infectious material).

    validate_args : :class:`bool`
        Passed to the
        :class:`~numpyro.distributions.distribution.Distribution`
        constructor to enable / disable
        parameter validation.
        Default :py:data:`True`.


    Returns
    -------
    dist : :class:`~pyter.distributions.TiterPlate`
        A :class:`~numpyro.distributions.distribution.Distribution`
        object representing the distribution of the
        well plaque counts (plaque assay) or
        positive / negative statuses
        (endpoint titration assay).

    """

    assays = {"tcid": EndpointTiterPlate, "pfu": PlaquePlate}

    if assay is None:
        raise ValueError("Must specify assay")

    distribution = assays.get(assay, None)

    if distribution is None:
        raise ValueError(
            "Unknown or unsupported "
            "assay {}.\n\n"
            "Supported assays are endpoint titration "
            "(set assay = 'tcid') and "
            "plaque assay (set assay = 'pfu')"
            "".format(assay)
        )
    return distribution(
        log_titer=log_titer,
        log_dilution=log_dilution,
        log_base=log_base,
        well_volume=well_volume,
        false_hit_rate=false_hit_rate,
        validate_args=validate_args,
    )


def validate_not_instantiated(distribution):
    if not (
        inspect.isclass(distribution) and issubclass(distribution, dist.Distribution)
    ):
        if isinstance(distribution, dist.Distribution):
            raise ValueError(
                "Expected a constructor (function) that builds "
                "numpyro Distributions, got an instantiated "
                "distribution. Did you type <DistributionName>() "
                "when you meant <DistributionName> (e.g. "
                " ``Normal()`` instead of ``Normal``)?"
            )
        else:
            raise ValueError(
                "Expected a constructor that builds "
                "numpyro Distributions, got {}"
                "".format(distribution)
            )
        pass
    return True


def sample_non_hier(
    param_name: str, param_dim: int, param_prior: dist.Distribution
) -> jax.Array:
    """
    Sample a vector of inferred
    parameters whose prior is fixed

    Convenience wrapper for :func:`numpyro.sample()
    <numpyro.primitives.sample>` to sample
    a vectorized parameter that is non-hierarchical.

    Parameters
    ----------
    param_name : :class:`str` :
         The name of the parameter

    param_dim : :class:`int` :
         The length of the parameter vector

    param_prior : :class:`~numpyro.distributions.distribution.Distribution`
         A prior distribution for the parameter

    Returns
    -------
    param : :class:`jax.Array`:
         The sampled parameter vector.

    """
    param = npro.sample(param_name, param_prior.expand((param_dim,)))
    return param


def sample_loc_scale_hier(
    param_name: str,
    param_dim: int,
    n_locs: int,
    n_scales: int,
    param_distribution: dist.Distribution,
    loc_ids: ArrayLike,
    scale_ids: ArrayLike,
    loc_prior: dist.Distribution,
    scale_prior: dist.Distribution,
) -> jax.Array:
    """
    Sample a vector of hierarchical inferred
    parameters alongside their inferred
    parent parameters.

    Convenience wrapper to sample
    a vectorized parameter in which individual
    values are "loc/scale" hierarchical. That is,
    parameter values have a distribution
    that is determined by two parameters---
    a location parameter
    (``loc``, e.g. the mean/median/mode of a
    :class:`~numpyro.distributions.continuous.Normal`
    distribution) scale parameter
    (``scale``, e.g. the standard
    deviation of a
    :class:`~numpyro.distributions.continuous.Normal`
    distribution)---but the values of the
    location and/or the scale parameter
    are unknown and inferred alongside the
    child parameters.

    Parameters
    ----------
    param_name : :class:`str` :
        The name of the parameter.

    param_dim : :class:`int` :
        The length of the parameter vector to sample.

    n_locs : :class:`int` :
        The number of groups of ``loc``
        (e.g. mean, mode) values across all the
        parameters in the vector, e.g. 3 groups
        of parameters where group members
        are Normally distributed about unknown
        means :math:`\\mu_1`, :math:`\\mu_2`,
        and :math:`\\mu_3`, respectively.

    n_scales : :class:`int` :
        The number of groups of ``scale``
        (e.g. standard deviation) values
        across all the parameters in the
        vector, e.g. 3 groups of parameters
        whose members are Normally distributed
        about their (possibly shared,
        see ``n_locs``) unknown means
        with unknown shared standard deviations
        :math:`\\sigma_1`, :math:`\\sigma_2`,
        and :math:`\\sigma_3` respectively.

    param_distribution : :class:`~numpyro.distributions.
    distribution.Distribution`
        A loc / scale parameterizable probability distribution.

    loc_ids : :data:`~numpy.typing.ArrayLike`
        Array of ids associating each parameter in the
        desired vector to one of the ``n_locs``
        location parameters to be inferred.

    scale_ids : :data:`~numpy.typing.ArrayLike`
        Array of ids associating each parameter in the
        desired vector to one of the ``n_scales``
        scale parameters to be inferred.

    loc_prior : :class:`~numpyro.distributions.distribution.Distribution`
        Prior distribution for the inferred unknown
        ``loc`` parameters.

    scale_prior : :class:`~numpyro.distributions.distribution.Distribution`
        Prior distribution for the inferred unknown
        ``scale`` parameters.


    Returns
    -------
    param : :class:`jax.Array`
        A sampled vector of parameters.

    """
    param_loc = npro.sample(param_name + "_loc", loc_prior.expand((n_locs,)))
    param_scale = npro.sample(param_name + "_scale", scale_prior.expand((n_scales,)))

    param = npro.sample(
        param_name,
        param_distribution(loc=param_loc[loc_ids], scale=param_scale[scale_ids]),
    )

    return param


def loc_scale_factory(
    distribution: str, loc: ArrayLike = None, scale: ArrayLike = None
) -> dist.Distribution:
    """Factory function for distributions
    with a loc/scale parameterization

    Parameters
    ----------
    distribution : :class:`str`
        the name of the desired distribution
    loc :  :data:`~numpy.typing.ArrayLike`
        the location parameter(s) of the desired distribution
    scale :  :data:`~numpy.typing.ArrayLike`
        the scale parameter(s) of the desired distribution

    Returns
    -------
    dist : :class:`~numpyro.distributions.distribution.Distribution`
        The parameterized distribution.

    """
    distributions = {
        "normal": dist.Normal,
        "cauchy": dist.Cauchy,
        "studentt": dist.StudentT,
    }
    dist_pick = distributions.get(distribution, None)
    if dist_pick is None:
        raise ValueError(
            "Unknown or unsupported "
            "distribution {}.\n\n"
            "Supported distributions: {}"
            "".format(distribution, [key for key in distributions.keys()])
        )
    return dist_pick(loc=loc, scale=scale)


@attrs.define
class AbstractModel:
    """
    Abstract base class for Pyter models
    """

    reparam_dict: dict = attrs.Factory(dict)

    def model(self, data: dict = None):
        """

        Parameters
        ----------
        data :
             (Default value = None)

        Returns
        -------

        """
        raise NotImplementedError()

    def get_reparam(self):
        """ """
        return reparam(self.model, self.reparam_dict)

    def validate_data(self, data: pdata.AbstractData, run_data: dict):
        """

        Parameters
        ----------
        data :
        run_data :

        Returns
        -------

        """
        raise NotImplementedError()

    def validate_model(self):
        raise NotImplementedError()

    def __call__(self, x):
        return self.model(data=x)


@attrs.define
class TiterModel(AbstractModel):
    """
    Model to infer individual titers independently
    """

    log_titer_prior: dist.Distribution = None
    assay: str = "tcid"

    def model(self, data: dict = None):
        """

        Parameters
        ----------
        data :

        Returns
        -------

        """

        log_titer = sample_non_hier(
            "log_titer", data["n_values"]["titer"], self.log_titer_prior
        )

        wells = npro.sample(
            "well_status",
            well_distribution_factory(
                assay=self.assay,
                log_titer=log_titer[data["well_internal_id_values"]["titer"]],
                log_dilution=data["well_dilution"],
                log_base=data["log_base"],
                well_volume=data["well_volume"],
                false_hit_rate=data["false_hit_rate"],
                validate_args=True,
            ),
            obs=data["well_status"],
        )

        return wells

    def validate_data(self, data: pdata.AbstractData, run_data: dict):
        """

        Parameters
        ----------
        data : :class:`~pyter.data.TiterData` :
            Pyter data object to validate.

        run_data : :class:`dict` :
            Frozen dictionary of data with which
            to fit the model, generated from
            a :class:`TiterData` object
            by the :meth:`~pyter.data.TiterData.freeze`
            method.

        Returns
        -------
        :py:data:`True`

        Raises
        ------

        """
        if not isinstance(data, pdata.TiterData):
            raise ValueError(
                "Incorrect data type {} for model {}".format(type(data), type(self))
            )
        return True


@attrs.define
class HalfLifeModel(AbstractModel):
    """
    Model to infer virus halflives from
    experimental timeseries data.

    A timeseries here is any set titration
    results taken at different timepoints
    that represent repeat samples from the
    same viral stock. But we can also handle
    cases in which non-destructive sampling
    is impossible (for example, depositing
    stock onto a surface and retrieving it at
    :math:`t = 0` h, :math:`t=1` h, etc.).
    To do this, we use a hierarchical approach:
    we infer a shared halflife for the samples
    jointly with a and modal value for the
    initial titer deposited. Each individual
    sample's unknown :math:`t = 0` value
    may vary about this value. This allows
    the model to use the immediately retrieved
    t = 0 titers to make inferences about the
    what the unmeasured :math:`t = 0` h titers
    were for the samples taken at :math:`t = 1` h,
    :math:`t = 2` h, etc. samples.
    """

    assay: str = "tcid"
    halflives_hier: bool = False
    intercepts_hier: bool = False
    titers_overdispersed: bool = False

    log_halflife_distribution: dist.Distribution = attrs.Factory(lambda: dist.Normal)
    log_halflife_loc_prior: dist.Distribution = None
    log_halflife_scale_prior: dist.Distribution = None

    log_intercept_distribution: dist.Distribution = attrs.Factory(lambda: dist.Normal)
    log_intercept_loc_prior: dist.Distribution = None
    log_intercept_scale_prior: dist.Distribution = None

    log_titer_error_distribution: dist.Distribution = attrs.Factory(lambda: dist.Normal)
    log_titer_error_scale_prior: dist.Distribution = None

    def __attrs_post_init__(self):
        for condition, param in zip(
            [
                self.halflives_hier,
                self.intercepts_hier,
                self.titers_overdispersed,
            ],
            ["log_halflife", "log_titer_intercept", "log_titer"],
        ):
            if condition:
                self.reparam_dict[param] = LocScaleReparam(0)

    def sample_log_halflife(self, data: dict = None) -> jax.Array:
        """
        Sample log half-life values, either
        from a fixed-parameter prior or hierarchically,
        as specified for the user.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of data with which to fit the model.
            Defaults to :py:data:`None`.


        Returns
        -------
        log_halflife: :class:`jax.Array`
            An array of sampled halflives.

        """
        if self.halflives_hier:
            log_halflife = sample_loc_scale_hier(
                "log_halflife",
                data["n_values"]["halflife"],
                data["n_values"]["halflife_loc"],
                data["n_values"]["halflife_scale"],
                self.log_halflife_distribution,
                data["halflife_internal_id_values"]["loc"],
                data["halflife_internal_id_values"]["scale"],
                self.log_halflife_loc_prior,
                self.log_halflife_scale_prior,
            )
        else:
            log_halflife = sample_non_hier(
                "log_halflife",
                data["n_values"]["halflife"],
                self.log_halflife_distribution,
            )

        return log_halflife

    def sample_log_titer_intercept(self, data: dict = None) -> jax.Array:
        """
        Sample log intercept (i.e. t = 0) values for the
        modeled titers, either a fixed-parameter prior
        or hierarchically, as specified for the user.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of data with which to fit the model.
            Defaults to :data:`None`.

        Returns
        -------
        log_titer_intercept : :class:`jax.Array`
            An array of sampled intercepts.

        """
        if self.intercepts_hier:
            log_titer_intercept = sample_loc_scale_hier(
                "log_titer_intercept",
                data["n_values"]["intercept"],
                data["n_values"]["intercept_loc"],
                data["n_values"]["intercept_scale"],
                self.log_intercept_distribution,
                data["intercept_internal_id_values"]["loc"],
                data["intercept_internal_id_values"]["scale"],
                self.log_intercept_loc_prior,
                self.log_intercept_scale_prior,
            )
        else:
            log_titer_intercept = sample_non_hier(
                "log_titer_intercept",
                data["n_values"]["intercept"],
                self.log_intercept_distribution,
            )
        return log_titer_intercept

    def sample_log_titer(
        self, predicted_titer: jax.Array, data: dict = None
    ) -> jax.Array:
        """
        Sample realized log titer values for the modeled
        titers, either deterministically predicted from
        the other parameters, or with an inferred degree
        of noise, as specified by the user.

        Parameters
        ----------
        predicted_titer : :class:`jax.Array`
            An array of predicted titer values.

        data : :class:`dict`
            Dictionary of data with which to fit the model.
            Defaults to :py:data:`None`.

        Returns
        -------
        log_titer : :class:`jax.Array`

        """
        if self.titers_overdispersed:
            log_titer_error_scale = sample_non_hier(
                "log_titer_error_scale",
                data["n_values"]["titer_error_scale"],
                self.log_titer_error_scale_prior,
            )
            es_id = data["titer_internal_id_values"]["titer_error_scale"]
            log_titer = npro.sample(
                "log_titer",
                self.log_titer_error_distribution(
                    loc=predicted_titer, scale=log_titer_error_scale[es_id]
                ),
            )
        else:
            log_titer = npro.deterministic("log_titer", predicted_titer)
        return log_titer

    def model(self, data: dict | None = None) -> tuple[jax.Array, jax.Array]:
        """

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of data with which to fit the model.
            Defaults to :py:data:`None`, in which case an
            empty dictionary is used.

        Returns
        -------
        log_titer, wells : :class:`tuple`
        ( :class:`jax.Array`, :class:`jax.Array` )
            Tuple of arrays containing sampled log
            titer values and sampled
            well statuses / plaque counts.
        """
        if data is None:
            data = {}

        well_titer_id = data["well_internal_id_values"]["titer"]
        titer_hl_id = data["titer_internal_id_values"]["halflife"]
        titer_intercept_id = data["titer_internal_id_values"]["intercept"]

        log_halflife = self.sample_log_halflife(data=data)
        log_titer_intercept = self.sample_log_titer_intercept(data=data)

        halflife = npro.deterministic("halflife", jnp.exp(log_halflife))
        decay_rate = npro.deterministic(
            "decay_rate", jnp.log(2) / (halflife * jnp.log(data["log_base"]))
        )
        initial_log_titer = npro.deterministic(
            "initial_log_titer", log_titer_intercept[titer_intercept_id]
        )

        predicted_log_titer = npro.deterministic(
            "predicted_log_titer",
            initial_log_titer
            - decay_rate[titer_hl_id] * data["titer_time"]
            + data["log_titer_change_other"],
        )

        log_titer = self.sample_log_titer(predicted_log_titer, data=data)

        wells = npro.sample(
            "well_status",
            well_distribution_factory(
                assay=self.assay,
                log_titer=log_titer[well_titer_id],
                log_dilution=data["well_dilution"],
                log_base=data["log_base"],
                well_volume=data["well_volume"],
                false_hit_rate=data["false_hit_rate"],
                validate_args=True,
            ),
            obs=data["well_status"],
        )

        return (log_titer, wells)

    def validate_data(self, data: pdata.AbstractData, run_data: dict):
        """

        Parameters
        ----------
        data :
        run_data :

        Returns
        -------

        """
        if not isinstance(data, pdata.HalfLifeData):
            raise ValueError(
                "Incorrect data type {} for model {}".format(type(data), type(self))
            )
        pass


@attrs.define
class MultiphaseHalfLifeModel(HalfLifeModel):
    """ """

    n_phases: int = 2
    assay: str = "tcid"
    halflives_hier: bool = False
    intercepts_hier: bool = False
    titers_overdispersed: bool = False

    log_halflife_distribution: dist.Distribution = attrs.Factory(lambda: dist.Normal)
    log_halflife_loc_prior: dist.Distribution = None
    log_halflife_scale_prior: dist.Distribution = None

    log_intercept_distribution: dist.Distribution = attrs.Factory(lambda: dist.Normal)
    log_intercept_loc_prior: dist.Distribution = None
    log_intercept_scale_prior: dist.Distribution = None

    log_titer_error_distribution: dist.Distribution = attrs.Factory(lambda: dist.Normal)
    log_titer_error_scale_prior: dist.Distribution = None

    log_halflife_offset_prior: dist.Distribution = None
    breakpoint_delta_prior: dist.Distribution = None

    def __attrs_post_init__(self):
        if self.n_phases < 2:
            raise ValueError(
                "Attempt to initialize "
                "a multi-phase half-life "
                "model with fewer than two "
                "phases; use HalfLifeModel"
                "for monophasic decay."
            )
        super().__attrs_post_init__()

    def sample_log_halflife(self, data: dict = None):
        """

        Parameters
        ----------
        data :
             (Default value = None)

        Returns
        -------

        """
        if self.halflives_hier:
            log_halflife_first = sample_loc_scale_hier(
                "log_halflife_first",
                data["n_values"]["halflife"],
                data["n_values"]["halflife_loc"],
                data["n_values"]["halflife_scale"],
                self.log_halflife_distribution,
                data["halflife_internal_id_values"]["loc"],
                data["halflife_internal_id_values"]["scale"],
                self.log_halflife_loc_prior,
                self.log_halflife_scale_prior,
            )
            n_offsets = data["n_values"]["halflife_loc"]
        else:
            log_halflife_first = sample_non_hier(
                "log_halflife_first",
                data["n_values"]["halflife"],
                self.log_halflife_distribution,
            )
            n_offsets = data["n_values"]["halflife"]

        with npro.plate("offsets", n_offsets):
            with npro.plate("phases", self.n_phases - 1):
                breakpoint_deltas = npro.sample(
                    "breakpoint_deltas", self.breakpoint_delta_prior
                )
                break_times = npro.deterministic(
                    "breakpoint_times",
                    # cumsum columns to get break times
                    # for each experiment
                    jnp.cumsum(breakpoint_deltas, axis=0),
                )

                log_halflife_offsets = npro.sample(
                    "log_halflife_offsets", self.log_halflife_offset_prior
                )
                pass
            pass

        offset_stack = jnp.vstack(
            [jnp.zeros((self.n_phases - 1, n_offsets)), log_halflife_offsets]
        )

        if self.halflives_hier:
            expand_ids = data["halflife_internal_id_values"]["loc"]
            offset_stack = offset_stack[::, expand_ids]

        log_halflife = npro.deterministic(
            "log_halflife", log_halflife_first + offset_stack
        )

        return log_halflife, break_times

    def model(self, data: dict = None):
        """

        Parameters
        ----------
        data :
             (Default value = None)

        Returns
        -------

        """

        log_halflife, break_times = self.sample_log_halflife(data=data)
        log_titer_intercept = self.sample_log_titer_intercept(data=data)

        halflife = npro.deterministic("halflife", jnp.exp(log_halflife))
        decay_rate = npro.deterministic(
            "decay_rate", jnp.log(2) / (halflife * jnp.log(data["log_base"]))
        )
        # breakpoint_deltas and break_times have
        # shape (n_phases - 1, n_halflives)
        # we add a first row of zeros
        start_times = jnp.vstack([jnp.zeros_like(break_times[0, ::]), break_times])

        well_titer_id = data["well_internal_id_values"]["titer"]
        if self.halflives_hier:
            titer_break_id = data["titer_internal_id_values"]["halflife_loc"]
        else:
            titer_break_id = data["titer_internal_id_values"]["halflife"]

        # titer_break_times has shape (n_phases - 1, n_titers)
        titer_break_times = break_times[::, titer_break_id]
        titer_start_times = start_times[::, titer_break_id]
        titer_decay_rates = decay_rate[::, titer_break_id]

        # add a titer_time row at the end
        # of titer_break_times
        possible_end_times = jnp.vstack([titer_break_times, data["titer_time"]])

        # cut off each phase at the
        # end of the phase or the
        # observation time, whichever
        # is smaller
        cutoff_end_times = jnp.where(
            possible_end_times < data["titer_time"],
            possible_end_times,
            data["titer_time"],
        )

        # how much time (possibly 0!)
        # did the sample actually
        # spend in each of the decay
        # phases
        phase_times = npro.deterministic(
            "phase_times",
            jnp.where(
                titer_start_times < cutoff_end_times,
                cutoff_end_times - titer_start_times,
                0,
            ),
        )

        total_decay = npro.deterministic(
            "total_decay", jnp.sum(phase_times * titer_decay_rates, axis=0)
        )

        # predicted titer has length
        # n_titers
        predicted_log_titer = npro.deterministic(
            "predicted_log_titer", log_titer_intercept - total_decay
        )

        log_titer = self.sample_log_titer(predicted_log_titer, data=data)

        wells = npro.sample(
            "well_status",
            well_distribution_factory(
                assay=self.assay,
                log_titer=log_titer[well_titer_id],
                log_dilution=data["well_dilution"],
                log_base=data["log_base"],
                well_volume=data["well_volume"],
                false_hit_rate=data["false_hit_rate"],
                validate_args=True,
            ),
            obs=data["well_status"],
        )

        return (log_titer, wells)
