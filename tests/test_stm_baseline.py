"""Post-processing checks: error RMS, covariance units and real STM propagation."""
import json
import numpy as np
import pytest
from halo_tracking.stm_baseline import running_rmse, rolling_rmse, position_uncertainty_km, run_stm_baseline
from halo_tracking.study import load_config


def test_rmse_windows_and_missing_epochs_are_not_filled():
    day=np.arange(5,dtype=float)
    errors=np.array([3.,4.,np.nan,0.,np.nan])
    np.testing.assert_allclose(running_rmse(errors),[3.,np.sqrt(12.5),np.nan,np.sqrt(25/3),np.nan],equal_nan=True)
    np.testing.assert_allclose(rolling_rmse(day,errors,window_days=2),[3.,np.sqrt(12.5),np.nan,0.,np.nan],equal_nan=True)
    with pytest.raises(ValueError,match='increasing'):
        rolling_rmse([1.,0.],[1.,1.])


def test_position_uncertainty_is_root_trace_in_km():
    L=384400.
    trace=np.array([3./L**2,np.nan])
    np.testing.assert_allclose(position_uncertainty_km(trace,L),[np.sqrt(3),np.nan],equal_nan=True)
    with pytest.raises(ValueError,match='nonnegative'):
        position_uncertainty_km([-1.],L)


def test_stm_run_saves_errors_and_postprocessed_uncertainty(tmp_path):
    cfg=load_config()
    arrays,summary=run_stm_baseline(cfg,tmp_path,days=.12)
    assert summary['completed_requested_span']
    assert summary['last_simulated_day']==pytest.approx(.12)
    assert not arrays['model_invoked'].any()
    assert np.isfinite(arrays['observer_navigation_velocity_error_m_s']).all()
    for label in ('observer','gateway'):
        expected=np.sqrt(arrays[label+'_covariance_trace_normalized'])*cfg['length_unit_km']
        np.testing.assert_allclose(arrays[label+'_position_uncertainty_km'],expected)
    expected=np.sqrt(np.mean(arrays['gateway_tracking_position_error_km']**2))
    assert summary['rmse']['gateway_position_km']==pytest.approx(expected)
    assert summary['rmse']['gateway_position_km']==pytest.approx(arrays['gateway_running_position_rmse_km'][-1])
    assert json.loads((tmp_path/'summary.json').read_text())['planned_days']==.12
