"""Pure delta-v targets and real narxpy training on split-isolated data."""
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
NARX = pytest.importorskip('narxpy').NARX
from halo_tracking.delta_v_prediction import delta_v_targets, usable_rows, fit_delta_v_predictor
from halo_tracking.neural_narx import exogenous
from halo_tracking.two_stage_narx import _load, fit_observer_policy
from halo_tracking.study import load_config


def test_unknown_burn_is_excluded_and_does_not_become_zero_supervision():
    d = dict(observer_maneuver_required=np.array([[False,True,True,False,False]]),
        observer_maneuver_valid=np.array([[True,True,False,False,True]]),
        observer_target_delta_v_rotating_m_s=np.array([[[0,0,0],[1,2,3],
            [np.nan]*3,[np.nan]*3,[0,0,0]]]))
    y,known,active = delta_v_targets(d)
    np.testing.assert_array_equal(known,[[True,True,False,True,True]])
    np.testing.assert_array_equal(y[0,1],[1,2,3])
    np.testing.assert_array_equal(active,[[False,True,False,False,False]])
    mask = usable_rows(np.zeros((1,5,11)),known,1)
    # The unknown current target and next teacher-feedback window are excluded.
    np.testing.assert_array_equal(mask,[[False,True,False,False,True]])


def test_delta_v_network_trains_with_exactly_three_outputs(tmp_path):
    ds=Path(__file__).resolve().parents[1]/'artifacts'/'neural_narx'/'teacher_training'
    cfg=load_config()
    cfg['narx']['closed_loop_measurement_cadence_s']=21600.
    baseline=fit_observer_policy(ds,cfg,tmp_path/'baseline.joblib')
    model,report,predictions,data,masks=fit_delta_v_predictor(ds,cfg,tmp_path/'narx',baseline,epochs=3,hidden=8)
    assert model['network'].d_y==3
    assert report['epochs_trained']==3
    assert predictions['test']['NARX recursive'].shape==data['test']['observer_target_delta_v_rotating_m_s'].shape
    assert np.isfinite(predictions['test']['NARX recursive']).all()
    assert not masks['test'][:,:4].any()
    train=_load(ds/'training.npz')
    np.testing.assert_allclose(model['x_mean'],exogenous(train).mean((0,1)))
    torch.manual_seed(cfg['narx']['seed'])
    initial=NARX(4,4,11,3,8,'Identity')
    assert not torch.allclose(initial.hl1.weight,model['network'].hl1.weight)
    saved=torch.load(tmp_path/'narx'/'delta_v_narx.pt',weights_only=True)
    torch.testing.assert_close(saved['state_dict']['hl2.weight'],model['network'].hl2.weight)
