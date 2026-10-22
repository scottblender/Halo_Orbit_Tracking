"""Check network updates and independent identification datasets."""
import numpy as np
from halo_tracking.study import load_config
from halo_tracking.identification import generate_dataset
from halo_tracking.network import observations
from halo_tracking.dynamics import propagate
from halo_tracking.gateway_comparison import _filter,spacecraft_measurement


def test_dataset_splits_and_measurement_masks(tmp_path):
    cfg=load_config();cfg['identification'].update(training_arcs=2,validation_arcs=1,test_arcs=1)
    cfg['simulation']['cadence_s']=86400.
    report=generate_dataset(cfg,tmp_path)
    initial=[]
    for name in report:
        with np.load(tmp_path/(name+'.npz')) as data:
            np.testing.assert_allclose(data['transition_targets'],data['truth_rotating_normalized'][:,1:])
            np.testing.assert_allclose(data['transition_inputs'],data['truth_rotating_normalized'][:,:-1])
            assert np.isnan(data['measured_angles_rad'][~data['sensor_visible']]).all()
            assert np.isfinite(data['measured_angles_rad'][data['sensor_visible']]).all()
            np.testing.assert_allclose(np.diff(data['times_s']),86400.)
            initial.extend(data['initial_states_normalized'])
    assert len(np.unique(initial,axis=0))==4
    other=tmp_path/'repeat';generate_dataset(cfg,other)
    with np.load(tmp_path/'training.npz') as a,np.load(other/'training.npz') as b:
        np.testing.assert_equal(a['measured_angles_rad'],b['measured_angles_rad'])


def test_network_update_matches_single_site_limit():
    cfg=load_config();times=np.array([0.,.002,.004]);cfg['sensors']=cfg['sensors'][:1]
    o=propagate(np.array(cfg['orbit']['state']),times,cfg['mu'])
    g=propagate(np.array(cfg['gateway']['state']),times,cfg['mu'])
    obs=observations(times,o,cfg,1)
    from halo_tracking.measurements import predict
    z=np.array([predict(t,x,cfg) for t,x in zip(times,o)])
    zg=np.array([spacecraft_measurement(t,x,y,cfg) for t,x,y in zip(times,o,g)])
    truth=dict(times=times,truth=o,gateway_truth=g,lso_available=np.ones(3,dtype=bool),gateway_available=np.ones(3,dtype=bool))
    a=_filter(truth,cfg,z,zg,False)
    truth['sensor_visible']=np.ones((3,1),dtype=bool)
    b=_filter(truth,cfg,z[:,None,:],zg,False)
    np.testing.assert_allclose(a['state'],b['state'])
    np.testing.assert_allclose(a['covariance'],b['covariance'])
