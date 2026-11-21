"""LSO measurement provenance, missing-data handling and parameter recovery."""
import numpy as np
import pytest
from halo_tracking import lso_stationkeeping as lso
from halo_tracking.stationkeeping import load_settings
from halo_tracking.lso_diagnostics import longest_run, residual_report


@pytest.fixture(scope='module')
def orbit():
    cfg,s=load_settings()
    s.update(samples_per_period=32,horizon_steps=4)
    return lso.OrbitModel(cfg,s)


def test_lso_stream_has_real_visibility_and_angular_noise(orbit):
    rec=lso.make_experiment(orbit,1,44)
    assert rec.angles.shape==(1,33,len(orbit.sites),2)
    assert rec.visible.any()
    assert np.isnan(rec.angles[~rec.visible]).all()
    assert np.isfinite(rec.y).all()  # coasts through gaps, not fabricated angles
    assert np.linalg.eigvalsh(rec.covariance).min()>-1e-9


def test_gains_fit_angles_without_reading_state_estimates_or_truth(orbit):
    rng=np.random.default_rng(91)
    gains=np.array([.94,1.07,.97]);u=rng.normal(0,.01,(3,orbit.n,3))
    angles=[]
    for commands in u:
        z=np.zeros(6);rows=[]
        for k in range(orbit.n+1):
            if k:z=orbit.A[k-1]@z+orbit.B[k-1]@(gains*commands[k-1])
            nominal,C=orbit.observation_terms(k)
            rows.append((nominal.ravel()+C@z).reshape(-1,2))
        angles.append(rows)
    fake=np.full((3,orbit.n+1,6),np.nan)
    rec=lso.Experiment(u,fake,fake,np.arange(3),np.asarray(angles),
                       np.ones((3,orbit.n+1,len(orbit.sites)),bool),
                       np.full((3,orbit.n+1,6,6),np.nan))
    fit=lso.fit_greybox(orbit,rec)
    np.testing.assert_allclose(fit['gains'],gains,atol=2e-3)


def test_hidden_angles_cannot_update_the_estimator(orbit):
    n=10;sites=len(orbit.sites)
    blank=np.full((1,n+1,6),np.nan)
    mask=np.zeros((1,n+1,sites),bool)
    raw=np.zeros((1,n+1,sites,2))
    rec=lso.Experiment(np.zeros((1,n,3)),blank,blank,np.array([1]),raw,mask,
                       np.zeros((1,n+1,6,6)))
    first=lso.predict(orbit,rec,np.ones(3))
    rec.angles[:]=1e8
    second=lso.predict(orbit,rec,np.ones(3))
    np.testing.assert_array_equal(first['navigation'],second['navigation'])
    np.testing.assert_array_equal(first['navigation'],np.zeros_like(first['navigation']))
    assert np.isnan(first['standardized']).all()


def test_gap_diagnostics_do_not_compress_time():
    rng=np.random.default_rng(11)
    e=rng.normal(size=(2,43,6));e[:,14:28]=np.nan
    q=residual_report(e,rng.normal(size=(2,42,3)),lags=4)
    assert q['eligible_whiteness_tests']==0
    assert q['contiguous_lengths'].max()<24
    assert len(longest_run(np.r_[np.ones(4),np.nan,np.ones(7)]))==7
