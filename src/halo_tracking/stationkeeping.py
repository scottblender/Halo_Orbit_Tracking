"""Station keeping only: nonlinear experiments, periodic STM, and grey-box PEM.

State/output convention: rotating-axis position error [km] and inertial
velocity error resolved on those axes [m/s]. Internal model states are divided
by the 10 km / 0.1 m/s corridor scales. Observations are PRE-impulse; u[k]
changes the velocity immediately AFTER y[k], before propagation to y[k+1].
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .dynamics import rhs, jacobian, propagate
from .orbit_reference import correct_halo_reference


@dataclass
class Experiment:
    """Independent arcs; never concatenate their boundaries for lag analysis."""
    u: np.ndarray                       # [arc, interval, xyz], commanded m/s
    y: np.ndarray                       # [arc, epoch, 6], noisy km / m/s
    truth: np.ndarray                   # same physical units, evaluation only
    seeds: np.ndarray


def load_settings(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    cfg = json.loads((root/'configs/baseline.json').read_text())
    settings = json.loads((root/'configs/stationkeeping.json').read_text())
    return cfg, settings


class OrbitModel:
    """Corrected nominal plus independent interval STMs; no accumulated inversion."""
    def __init__(self, cfg, settings):
        self.cfg = correct_halo_reference(cfg)
        self.settings = dict(settings)
        self.n = int(settings['samples_per_period'])
        self.T = self.cfg['orbit']['period_tu']
        self.dt = self.T/self.n
        self.dt_s = self.dt*self.cfg['time_unit_s']
        self.scale = np.repeat([settings['position_limit_km'],
                                settings['velocity_limit_m_s']], 3)
        L, TU = cfg['length_unit_km'], cfg['time_unit_s']
        W = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
        self.S = np.block([[L*np.eye(3), np.zeros((3, 3))],
                           [1000*L/TU*W, 1000*L/TU*np.eye(3)]])
        self.J = np.linalg.solve(self.S, np.diag(self.scale))
        self.Jinv = np.linalg.inv(self.J)
        x0 = np.asarray(self.cfg['orbit']['state'])
        sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']), (0, self.T), x0,
                        method='DOP853', rtol=4e-12, atol=4e-14, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        self._reference = sol.sol
        self.closure = self.S @ (sol.y[:, -1]-x0)
        matrices = []
        for k in range(self.n):
            def ode(t, flat):
                return (jacobian(self.reference(t), cfg['mu']) @
                        flat.reshape(6, 6)).ravel()
            z = solve_ivp(ode, (k*self.dt, (k+1)*self.dt), np.eye(6).ravel(),
                          method='DOP853', rtol=2e-10, atol=2e-12)
            if not z.success:
                raise RuntimeError(z.message)
            matrices.append(self.Jinv @ z.y[:, -1].reshape(6, 6) @ self.J)
        self.A = np.asarray(matrices)
        self.B = self.A[:, :, 3:] / self.scale[3:][None, None, :]
        self.R = np.diag((np.repeat([settings['navigation_position_sigma_km'],
                                     settings['navigation_velocity_sigma_m_s']], 3)
                          / self.scale)**2)
        # A small declared linearization-error floor, not a fitted noise model.
        self.Q = np.eye(6)*1e-14

    def reference(self, t):
        return self._reference(np.mod(t, self.T))

    def output(self, state, t):
        return self.S @ (np.asarray(state)-self.reference(t))

    def state(self, output, t):
        return self.reference(t) + np.linalg.solve(self.S, output)

    def step(self, state, k, command, gains=None, substeps=1, *, end_time=None):
        """Nonlinear plant, with a recorded pre/post impulse discontinuity."""
        gains = np.ones(3) if gains is None else np.asarray(gains)
        start = k*self.dt
        after = np.asarray(state).copy()
        after[3:] += (gains*np.asarray(command)*self.cfg['time_unit_s'] /
                      (1000*self.cfg['length_unit_km']))
        end = (k + 1) * self.dt if end_time is None else end_time
        times = np.linspace(start, end, substeps+1)
        states = propagate(after, times, self.cfg['mu'])
        return states[-1], times, states


def make_experiment(orbit, count, first_seed):
    """Open-loop, independent Gaussian commands: feedback cannot bias H1 here.

    Full nonlinear plant; independent full-state navigation noise. Commands are
    known exactly, while actuator gains differ from unity. Initial deviations
    and commands are drawn independently for each whole trajectory.
    """
    s = orbit.settings
    U, Y, truth, seeds = [], [], [], []
    sigma0 = np.repeat([s['initial_position_sigma_km'],
                        s['initial_velocity_sigma_m_s']], 3)
    sigma = np.sqrt(np.diag(orbit.R))*orbit.scale
    for seed in range(first_seed, first_seed+count):
        rng = np.random.default_rng(seed)
        u = rng.normal(0, s['excitation_sigma_m_s'], (orbit.n, 3))
        x = orbit.state(rng.normal(size=6)*sigma0, 0.)
        out = [orbit.output(x, 0.)]
        for k, command in enumerate(u):
            x, _, _ = orbit.step(x, k, command, s['simulated_actuator_gains'])
            out.append(orbit.output(x, (k+1)*orbit.dt))
        out = np.asarray(out)
        U.append(u)
        truth.append(out)
        Y.append(out+rng.normal(size=out.shape)*sigma)
        seeds.append(seed)
    return Experiment(np.asarray(U), np.asarray(Y), np.asarray(truth), np.array(seeds))


def prepare_study(root, output):
    """Deterministic rebuild; no reuse of old 100 km / 1 m/s teacher labels."""
    cfg, s = load_settings(root)
    orbit = OrbitModel(cfg, s)
    data = {}
    for i, name in enumerate(('training', 'validation', 'test')):
        data[name] = make_experiment(orbit, s[name+'_arcs'], s['seed']+1000*i)
    save_experiments(orbit, data, output)
    return orbit, data


def save_experiments(orbit, data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rec in data.items():
        np.savez_compressed(output/(name+'.npz'), u=rec.u, y=rec.y,
                            truth=rec.truth, seeds=rec.seeds)
    metadata = dict(configuration=orbit.cfg, settings=orbit.settings,
                    period_days=orbit.T*orbit.cfg['time_unit_s']/86400,
                    cadence_s=orbit.dt_s,
                    units=dict(u='m/s', y=['km']*3+['m/s']*3),
                    alignment='y[k] pre-burn; u[k] then coast; next output y[k+1]',
                    velocity_frame='inertial velocity error resolved in rotating axes',
                    closure_km_m_s=orbit.closure.tolist(),
                    split_seeds={n:r.seeds.tolist() for n,r in data.items()},
                    simulated=True, navigation='independent synthetic full-state observations',
                    sha256={n: hashlib.sha256((output/(n+'.npz')).read_bytes()).hexdigest()
                            for n in data})
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')


def restore_study(root, output):
    """Regenerate if absent; refuse a stale configuration or damaged split."""
    cfg, settings = load_settings(root)
    output = Path(output)
    if not (output/'experiment.json').exists():
        return prepare_study(root, output)
    meta = json.loads((output/'experiment.json').read_text())
    if meta['settings'] != settings:
        raise ValueError('Settings changed: rerun Notebook 01 to rebuild the data')
    orbit = OrbitModel(cfg, settings)
    if meta['configuration'] != orbit.cfg:
        raise ValueError('Orbit changed: rerun Notebook 01 to rebuild the data')
    data = {}
    for name, digest in meta['sha256'].items():
        p = output/(name+'.npz')
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Experiment checksum mismatch: '+name)
        with np.load(p, allow_pickle=False) as z:
            data[name] = Experiment(**{key:z[key] for key in ('u', 'y', 'truth', 'seeds')})
    return orbit, data


def filter_schedule(orbit, intervals, start=0, R=None):
    """Kalman covariance recursion; independent of the three fitted gains."""
    R = orbit.R if R is None else R
    P = R.copy()  # xhat[0]=y[0]; error covariance R, not zero
    K, chol = [], []
    for k in range(intervals):
        A = orbit.A[(start+k) % orbit.n]
        P = A @ P @ A.T + orbit.Q
        innovation_cov = P+R
        chol.append(np.linalg.cholesky(innovation_cov))
        gain = np.linalg.solve(innovation_cov, P.T).T
        K.append(gain)
        M = np.eye(6)-gain
        P = M @ P @ M.T + gain @ R @ gain.T  # Joseph form
    return np.asarray(K), np.asarray(chol)


def predict(orbit, rec, gains, free_run=False):
    """One-step predictions use past observations; free simulation uses y[0] only."""
    gains = np.asarray(gains)
    K, chol = filter_schedule(orbit, rec.u.shape[1])
    all_pred, all_eps, all_white = [], [], []
    for u, y in zip(rec.u, rec.y):
        state = y[0]/orbit.scale
        predictions, eps, white = [], [], []
        for k in range(len(u)):
            j = k % orbit.n
            prior = orbit.A[j] @ state + orbit.B[j] @ (gains*u[k])
            error = y[k+1]/orbit.scale-prior
            predictions.append(prior*orbit.scale)
            eps.append(error*orbit.scale)
            white.append(np.linalg.solve(chol[k], error))
            state = prior if free_run else prior+K[k] @ error
        all_pred.append(predictions)
        all_eps.append(eps)
        all_white.append(white)
    return dict(predicted=np.asarray(all_pred), residual=np.asarray(all_eps),
                standardized=np.asarray(all_white))


def fit_greybox(orbit, training):
    """Studio 11 PEM: Gaussian innovation likelihood, fixed physical STM and R.

    Only actuator gains are estimated. Since innovation covariance is fixed,
    minimizing whitened errors also minimizes the Gaussian negative likelihood.
    Synthetic plant gains never enter this estimator.
    """
    def errors(g):
        return predict(orbit, training, g)['standardized'].ravel()
    sol = least_squares(errors, np.ones(3), bounds=(.5, 1.5),
                        x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)
    sv = np.linalg.svd(sol.jac, compute_uv=False)
    return dict(gains=sol.x, gain_standard_errors=np.sqrt(np.diag(
                    np.linalg.pinv(sol.jac.T @ sol.jac))),
                jacobian_singular_values=sv, cost=float(sol.cost),
                optimizer_message=sol.message, fitted_parameters=3)


def prediction_scores(orbit, rec, gains):
    p = predict(orbit, rec, gains)
    f = predict(orbit, rec, gains, free_run=True)
    def score(e):
        return dict(position_rmse_km=float(np.sqrt(np.mean(np.sum(e[..., :3]**2, axis=-1)))),
                    velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum(e[..., 3:]**2, axis=-1)))))
    return dict(one_step=score(p['residual']), free_run=score(f['residual']))


def finite_horizon_diagnostics(orbit):
    """Phase sensitivity, MIMO coupling, Floquet multipliers and linearity test."""
    products = []
    for k in range(orbit.n):
        P = np.eye(6)
        for j in range(orbit.settings['horizon_steps']):
            P = orbit.A[(k+j) % orbit.n] @ P
        products.append(P)
    products = np.asarray(products)
    mono = np.eye(6)
    for A in orbit.A:
        mono = A @ mono
    scales = np.array([.01, .1, 1., 5.])
    phases = np.arange(0, orbit.n, max(1, orbit.n//8))
    mismatch = []
    direction = np.array([1., -.6, .4, .3, -.2, .1])
    for amplitude in scales:
        vals = []
        for k in phases:
            # amplitude is position magnitude scale [km]; velocity scaled consistently
            output = amplitude*direction*np.r_[np.ones(3), np.full(3, .01)]
            state = orbit.state(output, k*orbit.dt)
            for j in range(orbit.settings['horizon_steps']):
                state, _, _ = orbit.step(state, k+j, np.zeros(3))
            actual = orbit.output(state, (k+orbit.settings['horizon_steps'])*orbit.dt)/orbit.scale
            linear = products[k] @ (output/orbit.scale)
            vals.append(np.linalg.norm(actual-linear)/max(np.linalg.norm(actual), 1e-12))
        mismatch.append(vals)
    return dict(phase_deg=np.arange(orbit.n)*360/orbit.n,
                horizon_map=products,
                impulse_sensitivity=products[:, :, 3:]/orbit.scale[3:],
                multipliers=np.linalg.eigvals(mono),
                amplitudes_km=scales, linearity_relative_error=np.asarray(mismatch),
                sampled_linearity_phases_deg=phases*360/orbit.n)


def uncontrolled(orbit, periods=4):
    """Same small initial error as the controlled comparison; no state resets."""
    initial = np.array([.03, -.02, .01, .0003, -.0002, .0001])
    state = orbit.state(initial, 0.)
    outputs, times = [initial], [0.]
    for k in range(periods*orbit.n):
        state, tt, xx = orbit.step(state, k, np.zeros(3), substeps=4)
        outputs.extend(orbit.output(x, t) for x,t in zip(xx[1:], tt[1:]))
        times.extend(tt[1:])
    return np.asarray(times)*orbit.cfg['time_unit_s']/86400, np.asarray(outputs)


def plan_burn(orbit, estimate, k, gains):
    """Predict magnitude AND phase from current estimated departure, no schedule.

    Enumerate waits, solve a weighted path least-squares impulse, then validate
    each candidate with the nonlinear model. The current pre-burn state and all
    pre/post-burn samples are scored. Grid feasibility is not continuous proof.
    """
    s = orbit.settings
    horizon = s['horizon_steps']
    base = np.asarray(estimate)/orbit.scale
    limits = np.array([s['position_limit_km'], s['velocity_limit_m_s']])
    guesses = [(-1, np.zeros(3))]  # no burn
    for delay in s['candidate_delay_steps']:
        z, sensitivity = base.copy(), np.zeros((6, 3))
        offsets, matrices = [], []
        for j in range(horizon):
            A = orbit.A[(k+j) % orbit.n]
            z = A @ z
            sensitivity = A @ sensitivity
            if j == delay:
                sensitivity += orbit.B[(k+j) % orbit.n]*np.asarray(gains)[None, :]
            if j >= delay:
                offsets.append(z)
                matrices.append(sensitivity.copy())
        X, y = np.concatenate(matrices), np.concatenate(offsets)
        command = np.linalg.lstsq(np.vstack([X, .01*np.eye(3)]),
                                  np.r_[-y, np.zeros(3)], rcond=None)[0]
        if np.linalg.norm(command) <= s['maximum_impulse_m_s']:
            guesses.append((delay, command))
    candidates = []
    for delay, command in guesses:
        state = orbit.state(estimate, k*orbit.dt)
        path = [estimate]
        for j in range(horizon):
            u = command if j == delay else np.zeros(3)
            state, tt, xx = orbit.step(state, k+j, u, gains,
                                       substeps=s['nonlinear_samples_per_step'])
            path.extend(orbit.output(x, t) for x,t in zip(xx, tt))
        path = np.asarray(path)
        ratios = np.c_[np.linalg.norm(path[:, :3], axis=1)/limits[0],
                       np.linalg.norm(path[:, 3:], axis=1)/limits[1]]
        peak = float(ratios.max())
        # Prefer smallest impulse if margin is met; otherwise minimize predicted violation.
        excess = max(0., peak-s['planning_margin'])
        cost = excess**2 + s['fuel_weight']*np.linalg.norm(command)/s['maximum_impulse_m_s']
        candidates.append(dict(delay=delay, command=command, peak=peak, cost=cost))
    return min(candidates, key=lambda c:c['cost'])


def stationkeeping_trial(orbit, gains, periods=None, seed=26960):
    """Nonlinear closed loop with estimated state and autonomous candidate timing."""
    s = orbit.settings
    periods = s['stationkeeping_periods'] if periods is None else periods
    count = periods*orbit.n
    rng = np.random.default_rng(seed)
    initial = np.array([.03, -.02, .01, .0003, -.0002, .0001])
    state = orbit.state(initial, 0.)
    sigma = np.sqrt(np.diag(orbit.R))*orbit.scale
    estimate = initial+rng.normal(size=6)*sigma
    K, _ = filter_schedule(orbit, count)
    pending, scheduled = None, None
    outputs, days = [initial], [0.]
    epochs, commands, phases, plans = [], [], [], []
    for k in range(count):
        command = np.zeros(3)
        if pending is None:
            plan = plan_burn(orbit, estimate, k, gains)
            plans.append([k, plan['delay'], plan['peak']])
            if plan['delay'] >= 0 and np.linalg.norm(plan['command']) > 1e-10:
                pending, scheduled = plan['command'], k+plan['delay']
        if pending is not None and scheduled == k:
            command, pending = pending, None
            epochs.append(k*orbit.dt_s/86400)
            commands.append(command.copy())
            phases.append((k % orbit.n)*360/orbit.n)
        state, tt, xx = orbit.step(state, k, command, s['simulated_actuator_gains'],
                                   substeps=s['nonlinear_samples_per_step'])
        # Include both sides of impulses, retaining duplicate times.
        outputs.extend(orbit.output(x, t) for x,t in zip(xx, tt))
        days.extend(tt*orbit.cfg['time_unit_s']/86400)
        A, B = orbit.A[k % orbit.n], orbit.B[k % orbit.n]
        prior = A @ (estimate/orbit.scale)+B @ (np.asarray(gains)*command)
        observation = orbit.output(state, (k+1)*orbit.dt)+rng.normal(size=6)*sigma
        estimate = (prior+K[k] @ (observation/orbit.scale-prior))*orbit.scale
    y = np.asarray(outputs)
    p, v = np.linalg.norm(y[:, :3], axis=1), np.linalg.norm(y[:, 3:], axis=1)
    outside = (p > s['position_limit_km']) | (v > s['velocity_limit_m_s'])
    days = np.asarray(days)
    summary = dict(seed=seed, periods=periods, completed=True,
                   maximum_position_km=float(p.max()), maximum_velocity_m_s=float(v.max()),
                   all_evaluated_samples_inside=bool(not outside.any()),
                   first_sampled_exit_day=float(days[outside][0]) if outside.any() else None,
                   burns=len(commands), total_commanded_delta_v_m_s=float(
                       sum(np.linalg.norm(c) for c in commands)),
                   sampling='8 substeps/interval by default, including both sides of impulses; not a continuous guarantee',
                   navigation='synthetic noisy full-state; no Gateway or optical tracking')
    return dict(day=days, output=y, burn_day=np.asarray(epochs),
                commands=np.asarray(commands).reshape(-1, 3), phase_deg=np.asarray(phases),
                plans=np.asarray(plans), summary=summary)
