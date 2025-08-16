#!/usr/bin/env python3

# filename: data.py
# description: data structures
# for pyter inference

import attrs
import numpy as np
from numpy.typing import ArrayLike


def to_internal_ids(
    external_ids: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Internally index a long tidy
    data frame.

    Parameters
    ----------
    external_ids : :data:`~numpy.typing.ArrayLike`
        Array of external ids, which may be strings, numeric
        values, or another type coercible to a
        :class:`numpy.array`.


    Returns
    -------
    result : :class:`tuple`

        A tuple containing:


        **internal_ids** : :class:`numpy.ndarray`
            An array of the assigned internal id values for each entry
            ("row") of the provided external ids.

        **unique_internal_ids** : :class:`numpy.ndarray`
            An array of the unique values of the internal ids

        **unique_external_ids** : :class:`numpy.ndarray`
            An array of the unique values of the provided external ids

        **representative_rows** : :class:`numpy.ndarray`
            An array of the indices for the ``external_ids`` and
            ``internal_ids`` arrays that return one instance
            for each unique id value. This can then be used
            to index any other columns of the original data
            table from which the ``external_ids`` came.


    """
    (unique_external_ids, internal_ids) = np.unique(
        external_ids, return_inverse=True
    )

    (unique_internal_ids, representative_rows) = np.unique(
        internal_ids, return_index=True
    )

    n_values = unique_internal_ids.size

    return (
        internal_ids,
        unique_internal_ids,
        unique_external_ids,
        representative_rows,
        n_values,
    )


def validate_internal_ids(
    internal_ids: ArrayLike,
    unique_internal_ids: ArrayLike,
    unique_external_ids: ArrayLike,
    representative_rows: ArrayLike,
    n_values: int,
) -> None:
    """

    Parameters
    ----------
    internal_ids :
    unique_internal_ids:
    unique_external_ids :
        param representative_rows:
    n_values :

    unique_internal_ids :

    representative_rows :


    Returns
    -------

    """

    if not all(
        [
            representative_rows.size == n_values,
            unique_external_ids.size == n_values,
        ]
    ):
        raise ValueError("Inconsistent numbers of ids assigned")
    if n_values > 0:
        if not max(internal_ids) == n_values - 1:
            raise ValueError("Missing internal ids")
        if not np.all(np.sort(unique_internal_ids) == np.arange(n_values)):
            raise ValueError("Missing internal_ids")
        pass
    pass


def get_associated_internal_ids(
    key_param: str,
    value_param: str,
    internal_id_dict: dict,
    representative_row_dict: dict,
) -> np.ndarray:
    """For example, get internal intercept id
    for each internal titer id

    Parameters
    ----------
    key_param :
    value_param:
    internal_id_dict :
    representative_row_dict:

    Returns
    -------

    """
    rows = representative_row_dict[key_param]
    if rows.size > 0 and internal_id_dict[value_param].size > 0:
        return internal_id_dict[value_param][rows]
    else:
        return np.array([])


@attrs.define
class AbstractData:
    """
    Abstract base class for holding data associated
    to Pyter inferential models.
    """

    def freeze(self):
        """
        Validate, fix, and format data
        for use in inference.

        Data is returned as a :class:`dict`
        that can be passed to a
        corresponding :class:`Model
        <AbstractModel>` instance.

        The actual logic of validation
        and data preparation is handled
        by sub-class specific :meth:`validate`
        and :meth:`_freeze` methods; the
        common :meth:`freeze` method ensures
        common data dictionary output
        formatting across all
        :class:`Data <AbstractData>`
        subclasses.

        Returns
        -------
        data_dict : :class:`dict`
            A dictionary of data to pass to a model.

        """
        self.validate()
        self._freeze()
        return {
            s: getattr(self, s) for s in self.__slots__ if hasattr(self, s)
        }

    def _freeze(self):
        """ """
        raise NotImplementedError(
            "Abstract class AbstractDatahas no _freeze() method"
        )

    def validate(self):
        """ """
        raise NotImplementedError(
            "Abstract class AbstractDatahas no validate() method"
        )


@attrs.define
class NullData(AbstractData):
    """
    :class:`Data <AbstractData>` class for models that do not
    take any user-provided data, and
    for testing.
    """

    def _freeze(self):
        """
        Null data has no :meth:`_freeze` logic
        """
        pass

    def validate(self) -> bool:
        """
        Null data is necessarily valid

        Returns
        -------
        :data:`True`
        """
        return True


# TODO: template the internal id scheme?
@attrs.define
class TiterData(AbstractData):
    """
    :class:`Data <AbstractData>` class
    for inference of individual titers.
    """

    well_status: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_dilution: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_titer_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    log_base: np.ndarray = attrs.Factory(lambda: np.array([10]))
    well_volume: np.ndarray = attrs.Factory(lambda: np.array([1.0]))
    false_hit_rate: np.ndarray = attrs.Factory(lambda: np.array([0]))
    well_internal_id_values: dict = attrs.Factory(dict)
    unique_internal_ids: dict = attrs.Factory(dict)
    unique_external_ids: dict = attrs.Factory(dict)
    id_representative_rows: dict = attrs.Factory(dict)
    n_values: dict = attrs.Factory(dict)

    def _freeze(self):
        """
        create integer index and key from
        unique titer IDs (which may be numeric,
        character, etc)

        Parameters
        ----------

        Returns
        -------

        """
        (
            self.well_internal_id_values["titer"],
            self.unique_internal_ids["titer"],
            self.unique_external_ids["titer"],
            self.id_representative_rows["titer"],
            self.n_values["titer"],
        ) = to_internal_ids(self.well_titer_id)

    def validate(self) -> bool:
        """
        Null data is necessarily valid

        Returns
        -------
        :data:`True`
        """
        return True


@attrs.define
class HalfLifeData(AbstractData):
    """
    Data struct
    for inferring half-life
    of infectious virus
    """

    well_status: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_dilution: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_time: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_titer_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_titer_error_scale_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_halflife_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_halflife_loc_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_halflife_scale_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_intercept_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_intercept_loc_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    well_intercept_scale_id: np.ndarray = attrs.Factory(lambda: np.array([]))
    log_base: np.ndarray = attrs.Factory(lambda: np.array([10]))
    well_volume: np.ndarray = attrs.Factory(lambda: np.array([1.0]))
    false_hit_rate: np.ndarray = attrs.Factory(lambda: np.array([0]))

    well_internal_id_values: dict = attrs.Factory(dict)
    titer_internal_id_values: dict = attrs.Factory(dict)
    halflife_internal_id_values: dict = attrs.Factory(dict)
    intercept_internal_id_values: dict = attrs.Factory(dict)

    unique_internal_ids: dict = attrs.Factory(dict)
    unique_external_ids: dict = attrs.Factory(dict)
    id_representative_rows: dict = attrs.Factory(dict)
    n_values: dict = attrs.Factory(dict)

    titer_time: np.ndarray = None

    def update_internal_ids(self):
        """
        Assign internal ids
        for parameters
        """

        for param in [
            "titer",
            "titer_error_scale",
            "halflife",
            "halflife_loc",
            "halflife_scale",
            "intercept",
            "intercept_loc",
            "intercept_scale",
        ]:
            (
                self.well_internal_id_values[param],
                self.unique_internal_ids[param],
                self.unique_external_ids[param],
                self.id_representative_rows[param],
                self.n_values[param],
            ) = to_internal_ids(self.__getattribute__("well_" + param + "_id"))

        for param in [
            "halflife",
            "intercept",
            "titer_error_scale",
            "halflife_loc",
        ]:
            self.titer_internal_id_values[param] = get_associated_internal_ids(
                "titer",
                param,
                self.well_internal_id_values,
                self.id_representative_rows,
            )

        for param in ["loc", "scale"]:
            self.halflife_internal_id_values[param] = (
                get_associated_internal_ids(
                    "halflife",
                    "halflife_" + param,
                    self.well_internal_id_values,
                    self.id_representative_rows,
                )
            )
            self.intercept_internal_id_values[param] = (
                get_associated_internal_ids(
                    "intercept",
                    "intercept_" + param,
                    self.well_internal_id_values,
                    self.id_representative_rows,
                )
            )

        self.titer_time = self.well_time[self.id_representative_rows["titer"]]

    def index_prior_parameters(self):
        """
        Assign prior parameters to
        appropriate indices

        Parameters
        ----------

        Returns
        -------

        """
        pass

    def _freeze(self):
        """
        create integer index and key from
        unique titer IDs (which may be numeric,
        character, etc)

        Parameters
        ----------

        Returns
        -------

        """
        # TODO: args checking etc

        self.update_internal_ids()
        self.index_prior_parameters()

    def validate(self) -> bool:
        """
        Null data is necessarily valid

        Returns
        -------
        :data:`True`
        """
        return True
