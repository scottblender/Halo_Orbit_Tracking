"""Check Gateway/observer coupling and matched-comparison limits."""
from copy import deepcopy
import numpy as np
from numpy.testing import assert_allclose
from halo_tracking.study import load_config
from halo_tracking.dynamics import propagate
from halo_tracking.gateway_comparison import spacecraft_measurement,spacecraft_jacobians,_filter
from halo_tracking.estimation import run_ekf
from halo_tracking.measurements import predict,wrap


def short_truth(cfg):
    cfg['sensors']=cfg['sensors'][:1]
    times=np.array([0.,.002,.004])
    observer=propagate(np.array(cfg['orbit']['state']),times,cfg['mu'])
    gateway=propagate(np.array(cfg['gateway']['state']),times,cfg['mu'])
    return dict(times=times,truth=observer,gateway_truth=gateway,
                lso_available=np.ones(len(times),dtype=bool),gateway_available=np.ones(len(times),dtype=bool))


def test_spacecraft_measurement_jacobians():
    cfg=load_config();o=np.array(cfg['orbit']['state']);g=np.array(cfg['gateway']['state'])
    Ho,Hg=spacecraft_jacobians(.2,o,g,cfg)
    for is_observer,H in [(True,Ho),(False,Hg)]:
        for j in range(6):
            d=np.zeros(6);d[j]=1e-7
            plus=spacecraft_measurement(.2,o+d if is_observer else o,g if is_observer else g+d,cfg)
            minus=spacecraft_measurement(.2,o-d if is_observer else o,g if is_observer else g-d,cfg)
            diff=plus-minus;diff[0]=wrap(diff[0])
            assert_allclose(diff/(2e-7),H[:,j],rtol=2e-6,atol=1e-8)


def test_zero_observer_covariance_reduces_to_perfect_forecast():
    cfg=load_config();cfg['estimation']['initial_position_sigma_km']=0.;cfg['estimation']['initial_velocity_sigma_km_s']=0.
    truth=short_truth(cfg);empty=np.zeros((3,2))
    exact=_filter(truth,cfg,empty,empty,True,nominal=True)
    considered=_filter(truth,cfg,empty,empty,False,nominal=True)
    assert_allclose(exact['covariance'],considered['covariance'],rtol=1e-8,atol=1e-20)
    assert_allclose(considered['observer_covariance'],0,atol=1e-30)


def test_gateway_measurements_do_not_refine_observer():
    cfg=load_config();truth=short_truth(cfg)
    zl=np.array([predict(t,x,cfg) for t,x in zip(truth['times'],truth['truth'])])
    zg=np.array([spacecraft_measurement(t,o,g,cfg) for t,o,g in zip(truth['times'],truth['truth'],truth['gateway_truth'])])
    considered=_filter(truth,cfg,zl,zg,False)
    observer_cfg=deepcopy(cfg);observer_cfg['simulation']['seed']+=10
    standalone=run_ekf(truth['times'],truth['truth'],truth['lso_available'],zl,observer_cfg)
    assert_allclose(considered['observer_state'],standalone['state'],rtol=1e-10,atol=1e-13)
    assert_allclose(considered['observer_covariance'],standalone['covariance'],rtol=1e-7,atol=1e-20)
    for Po,Pog,Pg in zip(considered['observer_covariance'],considered['observer_gateway_covariance'],considered['covariance']):
        full=np.block([[Po,Pog],[Pog.T,Pg]])
        assert np.linalg.eigvalsh(full).min()>-1e-20
