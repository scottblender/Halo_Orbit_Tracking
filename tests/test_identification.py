"""Check network updates and independent identification datasets."""
import numpy as np
from halo_tracking.study import load_config
from halo_tracking.identification import generate_dataset
from halo_tracking.network import observations
from halo_tracking.dynamics import propagate
from halo_tracking.gateway_comparison import _filter,spacecraft_measurement


def test_stationkeeping_dataset_and_train_only_features(tmp_path):
    cfg=load_config()
    cfg['identification'].update(training_arcs=2, validation_arcs=1,
                                 test_arcs=1, decision_epochs_per_period=4)
    # Shorten optical sampling in CI without changing correction decisions.
    cfg['identification']['measurement_cadence_s']=86400.
    report=generate_dataset(cfg,tmp_path)
    initial=[]
    for name,item in report.items():
        assert item['decisions'] == item['arcs']*9
        assert item['triggered'] == item['valid_triggered']+item['rejected']
        assert item['no_burn']+item['triggered'] == item['decisions']
        with np.load(tmp_path/(name+'.npz')) as d:
            assert d['decision_navigation_states_normalized'].shape == (item['arcs'],9,6)
            assert d['decision_nominal_states_normalized'].shape == (item['arcs'],9,6)
            assert d['target_delta_v_rotating_m_s'].shape == (item['arcs'],9,3)
            assert d['target_delta_v_mci_m_s'].shape == (item['arcs'],9,3)
            np.testing.assert_allclose(
                np.diff(d['decision_times_tu']),
                d['decision_times_tu'][1]-d['decision_times_tu'][0])
            assert d['decision_navigation_covariance_diagonal_normalized'].shape == (item['arcs'],9,6)
            assert np.isnan(d['measured_angles_rad'][~d['sensor_visible']]).all()
            assert np.isfinite(d['measured_angles_rad'][d['sensor_visible']]).all()
            valid=d['maneuver_label_valid']
            burn=d['target_delta_v_rotating_m_s']
            assert np.isnan(burn[~valid]).all()
            assert np.isfinite(burn[valid]).all()
            assert np.all(np.linalg.norm(burn[valid],axis=1) <=
                          cfg['identification']['maximum_impulse_m_s']+1e-9)
            assert np.all(burn[~d['maneuver_required']] == 0.)
            accepted = valid & d['maneuver_required']
            assert np.all(
                d['predicted_with_burn_weighted_error'][accepted] <
                d['predicted_no_burn_weighted_error'][accepted])
            # The true state is saved for evaluation; teacher targets use
            # the separately estimated navigation state.
            assert not np.allclose(
                d['decision_navigation_states_normalized'],
                d['decision_truth_states_normalized'],rtol=0,atol=1e-14)
            initial.extend(d['initial_states_normalized'])
    assert len(np.unique(initial,axis=0)) == 4
    other=tmp_path/'repeat'
    generate_dataset(cfg,other)
    with np.load(tmp_path/'training.npz') as a,np.load(other/'training.npz') as b:
        np.testing.assert_equal(a['target_delta_v_rotating_m_s'],
                                b['target_delta_v_rotating_m_s'])
        np.testing.assert_equal(a['measured_angles_rad'],b['measured_angles_rad'])
    import json
    metadata=json.loads((tmp_path/'metadata.json').read_text())
    assert metadata['guidance_policy']['trigger_position_km'] == 50
    assert metadata['guidance_policy']['trigger_velocity_m_s'] == .5
    assert metadata['guidance_policy']['limit_position_km'] == 100
    assert metadata['guidance_policy']['limit_velocity_m_s'] == 1.


def test_guidance_trigger_and_infeasible_burn_policy():
    from halo_tracking.identification import _impulse_label
    from halo_tracking.orbit_reference import correct_halo_reference
    cfg=load_config()
    nominal_cfg=correct_halo_reference(cfg)
    state=np.asarray(nominal_cfg['orbit']['state'])
    setup=cfg['identification']
    policy={key:setup[key] for key in (
        'trigger_position_km','trigger_velocity_m_s',
        'limit_position_km','limit_velocity_m_s',
        'maximum_impulse_m_s','guidance_horizon_fraction',
        'regularization')}
    policy['period_tu']=nominal_cfg['orbit']['period_tu']
    nominal=lambda t:state
    # At t=0 the reference state agrees with the actual state: no burn.
    # The constant nominal function is only used at zero and a short
    # horizon for this no-burn/no-trigger unit test.
    nominal=lambda t:state
    no=_impulse_label(0.,state,nominal,cfg,policy)
    assert not no['trigger']
    assert no['feasible']
    np.testing.assert_array_equal(no['burn_rotating_m_s'],0.)
    # Extreme position displacement is outside a 1 cm/s burn budget.
    far=state.copy()
    far[0]+=2000/cfg['length_unit_km']
    policy['maximum_impulse_m_s']=.01
    rejected=_impulse_label(0.,far,nominal,cfg,policy)
    assert rejected['trigger']
    assert not rejected['feasible']
    assert np.isnan(rejected['burn_rotating_m_s']).all()


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
