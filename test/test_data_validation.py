#!/usr/bin/env python3

# filename: test_data_validation.py
# description: test that data validation
# implemented in pyter works as expected


import numpy as np
import pytest

from pyter.data import HalfLifeData, TiterData
from pyter.models import TiterModel


def test_model_data_mismatch_raises_error():
    titer_mod = TiterModel()

    hl_dat = HalfLifeData(
        well_titer_id=np.array(["titer1", "titer2"]),
        well_halflife_id=np.array(["hl1", "hl2"]),
        well_intercept_id=np.array(["hl1", "hl2"]),
        well_time=np.array([1.523, 1.595]),
    )

    with pytest.raises(ValueError, match="Incorrect data type"):
        frozen = hl_dat.freeze()
        titer_mod.validate_data(hl_dat, frozen)


def test_model_data_match_no_error():
    titer_dat_empty = TiterData()
    titer_mod = TiterModel()

    frozen = titer_dat_empty.freeze()
    titer_mod.validate_data(titer_dat_empty, frozen)
