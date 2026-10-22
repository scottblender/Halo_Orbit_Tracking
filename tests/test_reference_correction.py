"""Regression tests for periodic correction and optical range restrictions."""
from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose
from halo_tracking.orbit_reference import correct_halo_reference
from halo_tracking.study import load_config
from halo_tracking.dynamics import propagate
from halo_tracking.gateway_comparison import gateway_visible
from halo_tracking.network import observations


def test_corrected_periodic_reference_closes():
    cfg = load_config()
    reference = correct_halo_reference(cfg)
    assert reference['orbit']['reference_corrected']
    corrected = np.asarray(reference['orbit']['state'])
    T = reference['orbit']['period_tu']
    final = propagate(corrected, np.array([0., T]), reference['mu'])[-1]
    assert np.linalg.norm(final[:3] - corrected[:3])*cfg['length_unit_km'] < .001
    assert reference['orbit']['correction']['position_closure_km'] < .001
    assert np.linalg.norm(corrected - np.asarray(cfg['orbit']['state'])) > 0
    assert_allclose(correct_halo_reference(reference)['orbit']['state'], corrected)


def test_range_limits_remove_observations():
    cfg = load_config()
    state = np.asarray(cfg['orbit']['state'])
    far = state.copy()
    far[:3] += [3., 0., 0.]
    gateway = np.asarray(cfg['gateway']['state'])
    assert not gateway_visible(far, gateway, cfg)
    unlimited = deepcopy(cfg)
    unlimited['optics'].pop('gateway_max_range_km')
    assert gateway_visible(far, gateway, unlimited)
    cfg['optics']['lso_max_range_km'] = 1.
    net = observations(np.array([0.]), state.reshape(1, 6), cfg, 123)
    assert not net['sensor_visible'].any()
    assert np.isnan(net['measured_angles_rad']).all()
