"""Check reporting for each trajectory period and Gateway metric grouping."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from halo_tracking.study import load_config
from halo_tracking.departure import reference_departure, save_departure
from halo_tracking.plotting import departure_figures, period_trajectory_figures
from halo_tracking.gateway_comparison import gateway_period_metrics


def test_departure_per_period_plots_and_file(tmp_path):
    cfg=load_config()
    result=reference_departure(cfg,periods=3,samples_per_period=20)
    assert len(result['reference_period_states'])==21
    np.testing.assert_allclose(result['reference_period_states'][-1],result['states'][20],atol=1e-8)
    assert len(period_trajectory_figures(result,cfg))==3
    assert 'period_growth' in departure_figures(result,cfg)
    save_departure(result,cfg,tmp_path)
    with (tmp_path/'period_departure.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==3
    assert float(rows[0]['period'])==1
    plt.close('all')


def test_gateway_metric_period_boundaries():
    cfg=load_config()
    cfg['simulation']['periods']=2
    T=cfg['orbit']['period_tu']
    times=np.array([0,.25*T,T,1.25*T,2*T])
    truth=np.tile(np.asarray(cfg['gateway']['state']),(len(times),1))
    covariance=np.tile(np.eye(6)[None]*1e-12,(len(times),1,1))
    case=dict(state=truth+np.array([1e-6,0,0,0,0,0]),covariance=covariance)
    result=dict(config=cfg,times=times,gateway_truth=truth,
        gateway_available=np.array([1,1,0,1,0],dtype=bool),
        lso_available=np.array([1,0,1,0,1],dtype=bool),
        sensor_visible=np.array([[1,0],[0,0],[1,1],[0,1],[1,0]],dtype=bool),
        perfect=case,lso_estimated=case,perfect_forecast=case,lso_estimated_forecast=case)
    rows=gateway_period_metrics(result)
    assert [row['period'] for row in rows]==[1,1,2,2]
    assert [rows[0]['samples'],rows[2]['samples']]==[2,3]
    assert [rows[0]['gateway_observations'],rows[2]['gateway_observations']]==[2,1]
    assert [rows[0]['lso_observations'],rows[2]['lso_observations']]==[1,4]
    assert np.isfinite(rows[-1]['position_rmse_km'])
