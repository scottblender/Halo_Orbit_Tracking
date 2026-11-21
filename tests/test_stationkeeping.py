"""Scientific invariants for impulse alignment, frames, identification and statistics."""
import numpy as np
import pytest
from halo_tracking.stationkeeping import (OrbitModel, Experiment, load_settings,
                                          predict, fit_greybox, make_experiment)
from halo_tracking.stationkeeping_diagnostics import welch_bundle, residual_report, holm


@pytest.fixture(scope='module')
def orbit():
    cfg,s=load_settings()
    s.update(samples_per_period=32,horizon_steps=4)
    return OrbitModel(cfg,s)


def test_physical_frame_roundtrip_and_transport(orbit):
    q=np.array([2.,-3.,1.,.01,-.02,.03])
    np.testing.assert_allclose(orbit.output(orbit.state(q,.2),.2),q,atol=1e-10)
    offset=np.array([2.,-3.,1.,0.,0.,0.])
    normalized=np.r_[offset[:3]/orbit.cfg['length_unit_km'],np.zeros(3)]
    physical=orbit.S@normalized
    expected=1000/orbit.cfg['time_unit_s']*np.array([3.,2.,0.])
    np.testing.assert_allclose(physical[3:],expected,atol=1e-14)


def test_stm_impulse_matches_nonlinear_finite_difference(orbit):
    k=7
    h=1e-6
    for axis in range(3):
        command=np.eye(3)[axis]*h
        plus,_,_=orbit.step(orbit.reference(k*orbit.dt),k,command)
        minus,_,_=orbit.step(orbit.reference(k*orbit.dt),k,-command)
        derivative=orbit.S@(plus-minus)/(2*h)/orbit.scale
        np.testing.assert_allclose(derivative,orbit.B[k,:,axis],rtol=3e-3,atol=1e-3)


def test_greybox_recovers_hidden_actuation_without_truth_access(orbit):
    rng=np.random.default_rng(88)
    hidden=np.array([.94,1.07,.97])
    u=rng.normal(0,.001,(6,orbit.n,3))
    outputs=[]
    for commands in u:
        x=np.zeros(6)
        rows=[x*orbit.scale]
        for k,command in enumerate(commands):
            x=orbit.A[k]@x+orbit.B[k]@(hidden*command)
            rows.append(x*orbit.scale)
        outputs.append(rows)
    y=np.asarray(outputs)
    # Truth is deliberately unusable: fitting must rely only on u and y.
    rec=Experiment(u,y,np.full_like(y,np.nan),np.arange(6))
    fit=fit_greybox(orbit,rec)
    np.testing.assert_allclose(fit['gains'],hidden,atol=1e-5)
    pred=predict(orbit,rec,fit['gains'])
    assert np.max(abs(pred['residual']))<1e-6


def test_open_loop_generation_is_reproducible_and_splits_are_distinct(orbit):
    a=make_experiment(orbit,1,22)
    b=make_experiment(orbit,1,22)
    c=make_experiment(orbit,1,23)
    np.testing.assert_array_equal(a.y,b.y)
    assert not np.array_equal(a.u,c.u)
    assert a.y.shape[1]==a.u.shape[1]+1


def test_mimo_welch_recovers_cross_axis_gain_and_delay():
    rng=np.random.default_rng(9)
    u=rng.normal(size=(20,1024,3))
    mapping=np.array([[1.,.5,-.3],[-.2,2.,.8]])
    y=np.zeros((20,1024,2))
    y[:,1:]=u[:,:-1]@mapping.T
    spec=welch_bundle(u,y,1.,128)
    expected=np.exp(-2j*np.pi*spec['f'])[:,None,None]*mapping
    # Segment mean removal/window leakage biases the first nonzero bin of a
    # delay. Test the resolved interior, and check zero-delay cross-axis mixing
    # separately to machine precision, including the low-frequency bins.
    np.testing.assert_allclose(spec['H'][2:-1],expected[2:-1],atol=.025)
    instantaneous=welch_bundle(u,u@mapping.T,1.,128)
    np.testing.assert_allclose(instantaneous['H'],
                               np.broadcast_to(mapping,instantaneous['H'].shape),atol=1e-12)
    assert spec['nonoverlap_segments']==160
    assert spec['dof']==320


def test_residual_tests_detect_colored_noise_and_unexplained_input():
    rng=np.random.default_rng(777)
    u=rng.normal(size=(5,600,3))
    e=rng.normal(size=(5,600,6))
    for k in range(1,600):
        e[:,k,0]+=.8*e[:,k-1,0]
    e[:,:,1]=3*u[:,:,2]+.05*e[:,:,1]
    report=residual_report(e,u,12)
    assert np.all(report['ljung_box_holm_p'][:,0]<.01)
    assert report['ccf_holm_p'][0,1,2]<.01
    assert report['ccf_pairs'][12]==5*(600-12)
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
