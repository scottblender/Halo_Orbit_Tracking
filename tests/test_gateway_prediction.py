"""Blind-prediction truth/estimator separation and forecast-lifetime tests."""
from copy import deepcopy

import numpy as np

from halo_tracking.study import load_config
from halo_tracking.gateway_comparison import run_gateway_comparison
from halo_tracking.gateway_prediction import (
    run_gateway_prediction_study, save_gateway_prediction_study,
    load_gateway_prediction_study, time_since_last_observation, rolling_rmse)


def test_gap_clock_and_rolling_error():
    times = np.array([0., 1., 2., 4.])
    ages = time_since_last_observation(
        times, np.array([False, True, False, False]), 86400.)
    assert np.isnan(ages[0])
    np.testing.assert_allclose(ages[1:], [0., 1., 3.])
    values = rolling_rmse(times, np.array([1., 2., 3., 4.]), 2.)
    np.testing.assert_allclose(values, [1., np.sqrt(2.5),
                                        np.sqrt(14./3.), np.sqrt(12.5)])


def test_forced_truth_without_filter_acceleration():
    cfg = load_config()
    cfg['simulation']['cadence_s'] = 86400.
    ideal = run_gateway_comparison(cfg, duration_periods=1)
    loss = .55*cfg['orbit']['period_tu']
    perturbed = run_gateway_comparison(
        cfg, duration_periods=1, gateway_truth_acceleration_m_s2=2e-8,
        gateway_measurement_cutoff_tu=loss)
    np.testing.assert_allclose(ideal['times'], perturbed['times'])
    np.testing.assert_allclose(ideal['truth'], perturbed['truth'])
    delta = np.linalg.norm(ideal['gateway_truth'][-1, :3] -
                           perturbed['gateway_truth'][-1, :3])
    assert delta > 1e-10
    assert not np.any(perturbed['gateway_available'][perturbed['times'] > loss+1e-9])
    # This is a truth-only disturbance; no disturbance parameter is placed
    # in the CR3BP estimator configuration.
    assert 'gateway_truth_disturbance' not in perturbed['config']


def test_prediction_summaries_and_persistent_cache(tmp_path):
    cfg = load_config()
    cfg['simulation']['cadence_s'] = 21600.
    kwargs = dict(periods=2, blackout_period=1, disturbance_m_s2=2e-9,
                  thresholds_km=(1., 10.), rolling_window_days=1.)
    study = run_gateway_prediction_study(cfg, **kwargs)
    assert len(study['rows']) == 4
    assert len(study['threshold_rows']) == 4
    assert study['summary']['blackout_period'] == 1
    assert study['rows'][0]['horizon_periods'] == 1
    assert study['rows'][2]['horizon_periods'] == 2
    assert study['rows'][2]['gateway_observations_latest_period'] == 0
    assert (study['rows'][2]['gateway_observations_total'] ==
            study['rows'][0]['gateway_observations_total'])
    h = study['histories']
    assert np.isfinite(h['perfect_position_error_km']).all()
    assert np.isfinite(h['lso_estimated_rolling_position_rmse_km']).all()
    assert h['perfect_position_error_km'][-1] > 0
    assert np.nanmax(h['time_since_gateway_observation_days']) > 0
    assert not np.any(h['gateway_available'][h['times_tu'] >
                       study['summary']['blackout_period']*
                       study['summary']['nominal_period_tu']+1e-9])
    save_gateway_prediction_study(study, tmp_path)
    loaded = load_gateway_prediction_study(cfg, tmp_path, **kwargs)
    assert loaded is not None
    assert loaded['rows'][2]['gateway_observations_latest_period'] == 0
    np.testing.assert_allclose(
        loaded['histories']['perfect_position_error_km'],
        h['perfect_position_error_km'])
    assert load_gateway_prediction_study(
        cfg, tmp_path, periods=2, blackout_period=1,
        disturbance_m_s2=4e-9, thresholds_km=(1.,10.)) is None
    changed = deepcopy(cfg)
    changed['optics']['sun_exclusion_deg'] = 21.
    assert load_gateway_prediction_study(changed, tmp_path, **kwargs) is None


def test_invalid_prediction_request():
    cfg = load_config()
    import pytest
    with pytest.raises(ValueError, match='blackout_period'):
        run_gateway_prediction_study(cfg, periods=2, blackout_period=2)
    with pytest.raises(ValueError, match='Disturbance'):
        run_gateway_prediction_study(cfg, periods=2, blackout_period=1,
                                     disturbance_m_s2=0.)
