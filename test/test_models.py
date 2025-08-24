import pytest
import numpy as np
from numpyro.handlers import seed
import numpyro.distributions as dist
from pyter.models import HalfLifeModel
from pyter.data import HalfLifeData
import copy


@pytest.mark.parametrize(
    "data",
    [
        HalfLifeData(
            well_status=np.array([True, False, True, True]),
            well_titer_id=np.array(["titer_a", "titer_a", "titer_b", "titer_c"]),
            well_intercept_id=np.array(
                ["experiment_a", "experiment_a", "experiment_a", "experiment_b"]
            ),
            well_halflife_id=np.array(
                ["experiment_a", "experiment_a", "experiment_a", "experiment_b"]
            ),
            well_dilution=np.array([0, 0, 0, 0]),
            well_time=np.array([0.35, 5.23, 3, 0]),
        )
    ],
)
@pytest.mark.parametrize(
    "known_change_vec", [np.array([-1.5, 15.5, 2.6431]), np.array([-0.352]), 0, None]
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
            data_change = copy.deepcopy(data)
            data_change.log_titer_change_other = known_change_vec
        else:
            data_change = data
        sim_titers_change, sim_wells_change = model.model(data=data_change.freeze())
    assert all((sim_wells == 0) | (sim_wells == 1))
    assert all((sim_wells_change == 0) | (sim_wells_change == 1))
    if known_change_vec is not None:
        assert all(sim_titers + known_change_vec == sim_titers_change)
        assert not any(sim_titers + known_change_vec + 0.52 == sim_titers_change)
    else:
        assert all(sim_titers == sim_titers_change)
        assert not any(sim_titers + 0.52 == sim_titers_change)
