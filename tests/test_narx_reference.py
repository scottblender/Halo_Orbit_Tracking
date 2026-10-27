"""Phase wrapping, command decoding, and applied one-network station keeping."""
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('narxpy')
from halo_tracking.narx_reference import circular_phase_error, decode_predictions, fit_reference_narx, run_reference_pair
from halo_tracking.study import load_config


def test_phase_residual_wraps_across_zero():
    np.testing.assert_allclose(circular_phase_error([355,5,180],[5,355,0]),[-10,10,-180])


def test_command_decoding_rejects_instead_of_clipping():
    raw=np.zeros((3,8))
    raw[:,0]=[.7,.7,.2]
    raw[:,1]=[1,20,1]
    dv,burn,rejected=decode_predictions(raw,10.)
    np.testing.assert_allclose(dv,[[1,0,0],[0,0,0],[0,0,0]])
    np.testing.assert_array_equal(burn,[True,False,False])
    np.testing.assert_array_equal(rejected,[False,True,False])


def test_single_network_phase_predictions_and_matched_orbit_pair(tmp_path):
    cfg=load_config();cfg['narx']['closed_loop_measurement_cadence_s']=21600.
    ds=Path(__file__).resolve().parents[1]/'artifacts'/'neural_narx'/'teacher_training'
    bundle,stats=fit_reference_narx(ds,cfg,tmp_path/'prediction',epochs=3,hidden=8)
    assert stats['trained_networks']==1
    assert stats['epochs_trained']==3
    assert sum(stats['training_timing_counts'])==stats['training_burns']
    assert sum(stats['test_timing_counts'])==stats['reference_burns']
    with np.load(tmp_path/'prediction'/'prediction_residuals.npz') as r:
        np.testing.assert_allclose(r['delta_v_residual'],r['predicted_delta_v']-r['reference_delta_v'])
        phase=r['phase_residual_deg'][np.isfinite(r['phase_residual_deg'])]
        assert np.all((phase>=-180)&(phase<180))
        active=r['mask'] & r['reference_burn']
        measured=np.sqrt(np.mean(np.sum(r['delta_v_residual'][active]**2,-1)))
        assert np.isclose(measured,stats['burn_vector_rmse_m_s'])
    episodes,summary=run_reference_pair(cfg,bundle,tmp_path/'orbits',days=6.)
    assert set(episodes)=={'STM','NARX'}
    for name,e in episodes.items():
        assert e['completed_full_span']
        assert summary[name]['last_simulated_day']==pytest.approx(6.)
    assert episodes['NARX']['model_invoked'].any()
    assert episodes['NARX']['observer_truth_position_departure_km'][0]==episodes['STM']['observer_truth_position_departure_km'][0]
    cfg['narx']['history_steps']+=1
    with pytest.raises(ValueError,match='mismatch: history_steps'):
        run_reference_pair(cfg,bundle,tmp_path/'bad',days=1.)
