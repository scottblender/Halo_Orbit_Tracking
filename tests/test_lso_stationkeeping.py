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


def test_annual_trial_stops_at_requested_horizon_off_the_measurement_grid(orbit, monkeypatch):
    sample_days = orbit.dt_s / 86400
    duration = 2.5 * sample_days
    monkeypatch.setitem(orbit.settings, 'annual_horizon_days', duration)
    command = np.array([0.003, 0.004, 0.0])
    monkeypatch.setattr(lso.physics, 'plan_burn',
                        lambda *args, **kwargs: {'delay': 0, 'command': command})

    trial = lso.stationkeeping_trial(orbit, np.ones(3), duration_days=duration)

    assert trial['summary']['completed']
    assert trial['day'][-1] == pytest.approx(duration)
    assert np.all(trial['day'] <= duration + 1e-12)
    assert np.all(np.diff(trial['day']) >= 0)
    assert len(trial['angles']) == 3
    assert trial['summary']['annual_delta_v_m_s'] == pytest.approx(0.015)

    monkeypatch.setitem(orbit.settings, 'uncontrolled_stop_km', 0.0)
    truncated = lso.stationkeeping_trial(orbit, np.ones(3), duration_days=duration)
    assert not truncated['summary']['completed']
    assert truncated['summary']['annual_delta_v_m_s'] is None


def test_periodic_fit_recovers_synthetic_phase_gain_knots(orbit):
    knots = np.array([[.92, 1.05, .97], [1.04, .97, 1.09], [1.10, .94, .89], [.96, 1.10, 1.02]])
    phases = np.arange(orbit.n) / orbit.n
    gains = lso.periodic_phase_weights(phases, len(knots)) @ knots
    rng = np.random.default_rng(917)
    commands = rng.normal(0, .02, (4, orbit.n, 3))
    mask = np.ones((4, orbit.n + 1, len(orbit.sites)), dtype=bool)
    blank = np.full((4, orbit.n + 1, 6), np.nan)
    data = lso.Experiment(commands, blank, blank, np.arange(4),
                          np.zeros((*mask.shape, 2)), mask,
                          np.full((*blank.shape[:2], 6, 6), np.nan))
    for arc, u in enumerate(commands):
        state = np.zeros(6)
        for k in range(orbit.n + 1):
            if k:
                state = orbit.A[k-1] @ state + orbit.B[k-1] @ (gains[k-1] * u[k-1])
            nominal, C = orbit.observation_terms(k)
            data.angles[arc, k] = (nominal.ravel() + C @ state).reshape(-1, 2)
    result = lso.fit_periodic_greybox(orbit, data, windows=4, smoothing=0)
    np.testing.assert_allclose(result['knot_gains'], knots, atol=1e-5)
    weights = lso.periodic_phase_weights(np.array([-.01, .99, 1., 0.]), 4)
    np.testing.assert_allclose(weights[0], weights[1])
    np.testing.assert_allclose(weights[2], weights[3])
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    assert weights[1, -1] > 0 and weights[1, 0] > 0


def test_periodic_gain_is_used_at_the_executed_burn_phase(orbit, monkeypatch):
    selected = []

    def plan(_, estimate, k, gain_at, *, planning_margin):
        selected.append((gain_at(k + 1), planning_margin))
        return {'delay': 1, 'command': np.zeros(3)}

    monkeypatch.setattr(lso.physics, 'plan_burn', plan)
    gains = np.ones((orbit.n, 3))
    gains[1] = [1.1, .9, 1.05]
    lso.stationkeeping_trial(orbit, gains, duration_days=1.5 * orbit.dt_s / 86400)
    selected_gains, margin = selected[0]
    np.testing.assert_allclose(selected_gains, gains[1])
    assert margin == orbit.settings['controller_planning_margin']


def test_extended_navigation_updates_at_the_estimated_state(orbit):
    k = 3
    z = np.array([2.5, -1.3, .8, .2, -.1, .15])
    t = k * orbit.dt
    estimate = orbit.state(z * orbit.scale, t)
    measured = np.asarray([lso.angles_at(t, estimate, site) for site in orbit.sites])
    visible = np.ones(len(orbit.sites), dtype=bool)
    updated, covariance = lso._navigation_update(
        orbit, k, z, orbit.P0, measured, visible, 'extended')
    np.testing.assert_allclose(updated, z, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    linear, _ = lso._navigation_update(orbit, k, z, orbit.P0, measured, visible, 'linear')
    assert np.linalg.norm(linear - z) > 1e-6


def test_extended_navigation_accounts_for_burn_uncertainty(orbit, monkeypatch):
    z = np.zeros(6)
    command = np.array([.03, -.04, .02])
    gains = np.array([.95, 1.03, 1.08])
    monkeypatch.setitem(orbit.settings, 'controller_gain_sigma', 0.)
    predicted, known = lso._navigation_propagate(
        orbit, 0, z, orbit.P0, command, gains, 'extended')
    monkeypatch.setitem(orbit.settings, 'controller_gain_sigma', .10)
    repeated, uncertain = lso._navigation_propagate(
        orbit, 0, z, orbit.P0, command, gains, 'extended')

    end, _, _ = orbit.step(orbit.state(z, 0.), 0, command, gains)
    np.testing.assert_allclose(predicted, orbit.output(end, orbit.dt) / orbit.scale, atol=1e-8)
    np.testing.assert_allclose(predicted, repeated)
    assert np.trace(uncertain) > np.trace(known)
    assert np.linalg.eigvalsh(uncertain - known).min() >= -1e-12
