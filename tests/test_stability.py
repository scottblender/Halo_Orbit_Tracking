"""Check variational propagation against nonlinear perturbations and saved diagnostics."""
import json
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from halo_tracking.dynamics import propagate
from halo_tracking.orbit_reference import correct_halo_reference
from halo_tracking.stability import monodromy, orbit_stability, save_stability
from halo_tracking.study import load_config


def test_monodromy_matches_nonlinear_directional_derivative():
    cfg = correct_halo_reference(load_config())
    x = np.array(cfg['orbit']['state'])
    T = cfg['orbit']['period_tu']
    end, matrix = monodromy(x, T, cfg['mu'])
    direction = np.array([.3, -.2, .1, -.1, .4, -.3])
    epsilon = 1e-7
    plus = propagate(x+epsilon*direction, [0, T], cfg['mu'])[-1]
    minus = propagate(x-epsilon*direction, [0, T], cfg['mu'])[-1]
    assert_allclose(matrix@direction, (plus-minus)/(2*epsilon), rtol=2e-4, atol=2e-5)
    assert np.linalg.norm(end[:3]-x[:3])*cfg['length_unit_km'] < .001
    assert_allclose(np.linalg.det(matrix), 1., rtol=2e-6)


def test_stability_result_and_reproducible_ensemble(tmp_path):
    cfg = load_config()
    original = deepcopy(cfg)
    result = orbit_stability(cfg, periods=1, ensemble_count=2, samples_per_period=20)
    again = orbit_stability(cfg, periods=1, ensemble_count=2, samples_per_period=20)
    assert cfg == original
    assert_allclose(result['position_departure_km'], again['position_departure_km'])
    assert result['position_departure_km'].shape == (2, 21)
    assert result['summary']['classification'] == 'linearly unstable'
    assert result['summary']['spectral_radius'] > 1.01
    assert result['summary']['efolding_time_days'] > 0
    assert result['summary']['spectral_radius_tolerance_change'] < .001
    save_stability(result, tmp_path)
    summary = json.loads((tmp_path/'stability_summary.json').read_text())
    with np.load(tmp_path/'stability_histories.npz') as saved:
        assert_allclose(saved['monodromy'], result['monodromy'])
        outside = ((saved['position_departure_km'] >= summary['position_limit_km']) |
                   (saved['velocity_departure_m_s'] >= summary['velocity_limit_m_s']))
        assert summary['arcs_with_sampled_corridor_exit'] == outside.any(axis=1).sum()
        for i, day in enumerate(summary['first_sampled_exit_days']):
            if day is None:
                assert not outside[i].any()
            else:
                j = np.flatnonzero(outside[i])[0]
                assert_allclose(day, saved['times_tu'][j]*cfg['time_unit_s']/86400)


@pytest.mark.parametrize('settings', [dict(periods=0), dict(ensemble_count=0),
    dict(samples_per_period=2), dict(position_limit_km=-1), dict(velocity_spread_m_s=np.nan)])
def test_stability_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        orbit_stability(load_config(), **settings)
