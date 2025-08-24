import pytest
import numpy as np
from numpyro.handlers import seed
import numpyro.distributions as dist
from pyter.models import HalfLifeModel
from pyter.data import HalfLifeData


@pytest.mark.parametrize(
    ["data", "known_change_vec"],
    [
        [
            HalfLifeData(
                well_status=np.array([True]),
                well_titer_id=np.array(["titer_a"]),
                well_intercept_id=np.array(["intercept_a"]),
                well_halflife_id=np.array(["experiment_a"]),
                well_time=np.array([1]),
            ),
            np.array([-1.5]),
        ],
        [
            HalfLifeData(
                well_status=np.array([True]),
                well_titer_id=np.array(["titer_a"]),
                well_intercept_id=np.array(["intercept_a"]),
                well_halflife_id=np.array(["experiment_a"]),
                well_time=np.array([73]),
            ),
            np.array([0.25]),
        ],
        [
            HalfLifeData(
                well_status=np.array([True]),
                well_titer_id=np.array(["titer_a"]),
                well_intercept_id=np.array(["intercept_a"]),
                well_halflife_id=np.array(["experiment_a"]),
                well_time=np.array([0.35]),
            ),
            None,
        ],
    ],
)
def test_non_inactivation_change(data, known_change_vec):
    """
    Test that the log_titer_change_other argument
    works as expected for modeling known change in
    titers due to factors other than inactivation.
    """
    model = HalfLifeModel(
        log_halflife_distribution=dist.LogNormal(0, 1),
        log_intercept_distribution=dist.Normal(-1, 2),
    )
    with seed(rng_seed=5):
        sim_titers, sim_wells = model.model(data=data.freeze())
    with seed(rng_seed=5):
        if known_change_vec is not None:
            data.log_titer_change_other = known_change_vec
        sim_titers_change, sim_wells_change = model.model(data=data.freeze())
    assert all(sim_wells == 0 | sim_wells == 1)
    assert all(sim_wells_change == 0 | sim_wells_change == 1)
    if known_change_vec is not None:
        assert all(sim_titers + known_change_vec == sim_titers_change)
        assert not any(sim_titers + known_change_vec + 0.52 == sim_titers_change)
    else:
        assert all(sim_titers == sim_titers_change)
        assert not any(sim_titers + 0.52 == sim_titers_change)
