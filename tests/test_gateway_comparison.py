"""Check Gateway/observer coupling and matched-comparison limits."""
from copy import deepcopy
import numpy as np
from numpy.testing import assert_allclose
from halo_tracking.study import load_config
from halo_tracking.dynamics import propagate
from halo_tracking.gateway_comparison import spacecraft_measurement,spacecraft_jacobians,_filter,run_gateway_comparison,run_gateway_period_study,save_gateway_period_study,load_gateway_period_study
from halo_tracking.gateway_reference import GatewayReference
from halo_tracking.orbit_reference import correct_halo_reference
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


def test_gateway_reference_repeats_and_closes():
    cfg=load_config()
    reference=GatewayReference(cfg)
    assert reference.position_closure_km < .001
    assert reference.velocity_closure_m_s < .001
    phase=np.array([.1,.7,1.2])
    assert_allclose(reference.states(phase),reference.states(phase+reference.period),
                    atol=1e-13,rtol=1e-12)
    assert_allclose(reference.states([0])[0],cfg['gateway']['state'],atol=1e-13)


def test_gateway_period_comparison_is_standardized(tmp_path):
    cfg=load_config()
    cfg['simulation']['cadence_s']=21600.
    study=run_gateway_period_study(cfg,periods=2)
    rows=study['rows']
    assert len(rows)==4
    assert [row['horizon_periods'] for row in rows]==[1,1,2,2]
    assert [row['case'] for row in rows]==['perfect','lso_estimated']*2
    assert study['method']=='continuous_nominal_reference_cumulative_v4'
    assert study['observer_period_tu']==study['tracking_window_tu']
    assert np.isclose(study['tracking_window_tu'],
                      correct_halo_reference(cfg)['orbit']['period_tu'])
    # No per-period percentage change column; RMSE now aggregates from t=0.
    assert rows[2]['terminal_observer_departure_km']>0.
    assert rows[0]['terminal_observer_departure_km'] > 1.
    assert study['observer_reference_closure']['position_closure_km'] < 1e-3
    assert rows[0]['max_observer_departure_km'] >= 0.
    assert rows[2]['max_observer_departure_km'] >= rows[0]['max_observer_departure_km']
    assert rows[2]['gateway_observations'] >= rows[0]['gateway_observations']
    assert rows[2]['lso_observations'] >= rows[0]['lso_observations']
    assert (rows[2]['new_gateway_observations'] ==
            rows[2]['gateway_observations']-rows[0]['gateway_observations'])
    assert (rows[2]['new_lso_observations'] ==
            rows[2]['lso_observations']-rows[0]['lso_observations'])
    assert rows[0]['terminal_gateway_range_km'] > 0
    assert rows[2]['terminal_gateway_range_km'] > 0
    # Endpoint departure and relative Gateway range measure different geometries.
    assert not np.isclose(rows[2]['terminal_observer_departure_km'],
                          rows[2]['terminal_gateway_range_km'])
    assert all(np.isfinite(row['position_rmse_km']) for row in rows)
    save_gateway_period_study(study,tmp_path)
    assert (tmp_path/'gateway_period_metrics.csv').is_file()
    saved=load_gateway_period_study(cfg,tmp_path,periods=2)
    assert saved is not None and len(saved['rows'])==4
    assert saved['rows'][0]['position_rmse_km']==rows[0]['position_rmse_km']
    assert saved['rows'][2]['horizon_periods']==2
    assert saved['method']=='continuous_nominal_reference_cumulative_v4'
    assert saved['rows'][2]['terminal_gateway_range_km']==rows[2]['terminal_gateway_range_km']
    assert load_gateway_period_study(cfg,tmp_path,periods=3) is None
    altered=deepcopy(cfg)
    altered['measurement']['angle_sigma_arcsec']=2.
    assert load_gateway_period_study(altered,tmp_path,periods=2) is None
    # Old, independently reset CSVs must never be silently reused.
    import json
    summary_path=tmp_path/'gateway_period_summary.json'
    summary=json.loads(summary_path.read_text())
    summary.pop('method')
    summary_path.write_text(json.dumps(summary))
    assert load_gateway_period_study(cfg,tmp_path,periods=2) is None


def test_gateway_geometry_uses_full_observer_period():
    cfg=load_config()
    cfg['simulation']['cadence_s']=86400.
    result=run_gateway_comparison(cfg)
    assert np.isclose(result['times'][0],0.)
    assert np.isclose(result['times'][-1],cfg['orbit']['period_tu'])
    assert np.isclose(result['config']['simulation']['periods'],1.)
    reference=GatewayReference(cfg)
    assert_allclose(result['gateway_truth'],
                    reference.states(result['times']),atol=1e-12)
    from halo_tracking.plotting import geometry
    import matplotlib.pyplot as plt
    fig=geometry(result)
    assert len(fig.axes[0].lines)>=2
    plt.close(fig)
