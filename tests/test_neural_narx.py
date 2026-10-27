"""Neural-delay causality, library equivalence, and nonlinear integration checks."""
import numpy as np
import pytest
torch = pytest.importorskip('torch')
NARX = pytest.importorskip('narxpy').NARX
from halo_tracking.neural_narx import network_step, NeuralController, fit_neural_policy, targets, sequence, exogenous, load_neural_policy
from halo_tracking.closed_loop_narx import generate_closed_loop_dataset
from halo_tracking.narx_audit import audit_dataset
from halo_tracking.study import load_config
from halo_tracking.two_stage_narx import fit_observer_policy
from halo_tracking.year_study import run_paired_study


def test_online_matches_narxpy_and_strictly_past_output_feedback():
    torch.manual_seed(4)
    model = NARX(2,2,11,8,12,'Identity')
    x,y = torch.randn(1,7,11),torch.randn(1,7,8)
    actual=model(x,mode='open',y=y)
    expected=network_step(model,x[:,2:4],y[:,2:4])
    torch.testing.assert_close(actual[:,4],expected)
    altered=y.clone(); altered[:,4:]+=100
    torch.testing.assert_close(model(x,mode='open',y=altered)[:,4],actual[:,4])
    altered=y.clone(); altered[:,3]+=10
    assert not torch.allclose(model(x,mode='open',y=altered)[:,4],actual[:,4])
    # Current convention x[k-L+1:k+1] agrees with shifted library sequence.
    b=dict(network=model,history_steps=2,x_mean=np.zeros(11),x_scale=np.ones(11),
           y_mean=np.zeros(8),y_scale=np.ones(8),candidate_delay_fractions=[0,.0625,.125,.1875])
    online=NeuralController(b)
    inputs=x.numpy()[0]
    for row in inputs: online.command(row,1.,100.)
    recursive,_=sequence(b,inputs[None],np.zeros((1,7,8)),False)
    np.testing.assert_allclose(online.outputs,recursive.detach().numpy()[0],atol=1e-6)


def test_training_audit_and_four_policy_nonlinear_rollout(tmp_path):
    cfg=load_config()
    cfg['narx'].update(training_arcs=2,validation_arcs=1,test_arcs=1,
        arc_duration_periods=.5,history_steps=1,closed_loop_measurement_cadence_s=21600.)
    cfg['narx']['observer_guidance'].update(trigger_position_km=.0001,trigger_velocity_m_s=.0001)
    ds=tmp_path/'data'
    generate_closed_loop_dataset(cfg,ds)
    audit_dataset(ds,cfg)
    neural,report=fit_neural_policy(ds,cfg,tmp_path/'neural',epochs=8,hidden=12)
    loaded = load_neural_policy(tmp_path/'neural'/'neural_narx.pt')
    for name,value in neural['network'].state_dict().items():
        torch.testing.assert_close(value,loaded['network'].state_dict()[name])
    assert report['epochs_trained']==8
    assert report['feasible_training_burns']>0
    baseline=fit_observer_policy(ds,cfg,tmp_path/'baseline.joblib')
    result=run_paired_study(cfg,tmp_path/'year',baseline,days=2.,arcs=1,neural_bundle=neural)
    assert set(result['controllers'])=={'uncontrolled','stm','narx','neural_narx'}
    with np.load(tmp_path/'year'/'neural_narx_arc_000.npz') as d:
        assert d['model_invoked'][1]
        assert np.isfinite(d['observer_truth_position_departure_km']).all()
        assert d['delta_v_cost_m_s'].sum()>=0
    # Cadence and history mismatches cannot silently pass.
    with pytest.raises(ValueError,match='cadence mismatch'):
        run_paired_study(cfg,tmp_path/'bad',baseline,days=1,measurement_cadence_s=3600)
    cfg['narx']['history_steps']=2
    with pytest.raises(ValueError,match='history length'):
        run_paired_study(cfg,tmp_path/'bad',baseline,days=1)
    # Modifying only held-out data cannot alter training-only normalizers.
    with np.load(ds/'training.npz') as d:
        raw=exogenous(d)
        np.testing.assert_allclose(neural['x_mean'],raw.mean((0,1)))
        assert np.isfinite(targets(d,cfg)).all()


def test_delta_v_counts_separate_opposing_burns(monkeypatch):
    import halo_tracking.year_study as year
    from halo_tracking.gateway_reference_periodic import gateway_periodic_reference
    cfg=load_config()
    ref,T,corrected=year._observer_reference(cfg)
    gateway,_=gateway_periodic_reference(cfg)
    grid=np.array([0.,.05,.1])
    initial=year._initial_conditions(cfg,np.asarray(corrected['orbit']['state']),
        gateway(0),5,len(grid),len(cfg['sensors']))
    def command(t,*args):
        return dict(required=True,valid=True,burn=np.array([1. if t==0 else -1.,0,0]),
                    delay_tu=.03 if t==0 else 0.)
    monkeypatch.setattr(year,'_optimal_phase',command)
    e=year.simulate_one_arc(cfg,'stm',ref,T,gateway,grid,grid,initial)
    assert e['completed_full_span']
    np.testing.assert_allclose(e['applied_delta_v_rotating_m_s'][1],0.)
    assert e['delta_v_cost_m_s'].sum()==2.
    assert e['executed_burn_count'].sum()==2
