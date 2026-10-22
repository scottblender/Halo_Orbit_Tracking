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
    # Isolate the range gate by using an unobstructed target direction.
    # Removing a range cap cannot override Earth/Sun exclusion or occultation.
    from halo_tracking.visibility import optical_visible
    site = np.array([1-cfg['mu'], 0., 0.])
    near = site + np.array([0.0, 0.0, 0.2])
    far_target = site + np.array([0.0, 0.0, 1.0])
    unlimited = deepcopy(cfg)
    unlimited['optics']['gateway_max_range_km'] = None
    # Disable configured keep-outs specifically for this range-only test;
    # physical occultation still applies.
    unlimited['optics']['earth_exclusion_deg'] = 0.
    unlimited['optics']['sun_exclusion_deg'] = 0.
    assert optical_visible(near, far_target, 0., unlimited, link='gateway')
    limited = deepcopy(unlimited)
    limited['optics']['gateway_max_range_km'] = 250000.
    assert not optical_visible(near, far_target, 0., limited, link='gateway')
    cfg['optics']['lso_max_range_km'] = 1.
    net = observations(np.array([0.]), state.reshape(1, 6), cfg, 123)
    assert not net['sensor_visible'].any()
    assert np.isnan(net['measured_angles_rad']).all()


def test_sun_and_earth_keepout_boundaries():
    from halo_tracking.visibility import body_clear, optical_visible
    cfg = load_config()
    # Check a synthetic LOS/body geometry independent of ephemeris phase.
    observer = np.array([0., 0., 0.])
    earth = np.array([10., 0., 0.])
    target = lambda degrees: 20.*np.array([
        np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees)), 0.])
    radius = 0.01
    assert not body_clear(observer, target(9.9), earth, radius, 10.)
    assert body_clear(observer, target(10.), earth, radius, 10.)
    assert body_clear(observer, target(10.1), earth, radius, 10.)
    assert not body_clear(observer, target(19.9), earth, radius, 20.)
    assert body_clear(observer, target(20.), earth, radius, 20.)
    # Physical occultation applies even when the configured angle is zero.
    assert not body_clear(observer, target(0.), earth, 1., 0.)
    assert body_clear(observer, target(30.), earth, 1., 0.)
    assert cfg['optics']['sun_exclusion_deg'] == 20.
    assert cfg['optics']['earth_exclusion_deg'] == 10.
