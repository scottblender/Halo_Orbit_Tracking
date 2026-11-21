"""Active station-keeping workflow: raw LSO angles, visibility and state estimation.

The grey box is fitted to angular innovations, not to EKF states treated as
independent observations. Truth is used only for simulating sensors/evaluation.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import stationkeeping as physics
from .measurements import predict as angles_at, measurement_jacobian, wrap, ARCSEC
from .network import observations, sensor_configs


@dataclass
class Experiment(physics.Experiment):
    angles: np.ndarray                 # [arc,epoch,site,alpha/delta], rad; NaN in gaps
    visible: np.ndarray                # [arc,epoch,site]
    covariance: np.ndarray             # [arc,epoch,6,6], estimated physical covariance


class OrbitModel(physics.OrbitModel):
    def __init__(self, cfg, settings):
        super().__init__(cfg, settings)
        # Uniform orbit-phase sampling for this experiment, explicitly recorded.
        self.cfg['simulation']['cadence_s'] = self.dt_s
        self.sites = sensor_configs(self.cfg)
        self.sigma_angle = cfg['measurement']['angle_sigma_arcsec']*ARCSEC
        prior = np.repeat([settings['lso_prior_position_sigma_km'],
                           settings['lso_prior_velocity_sigma_m_s']], 3)/self.scale
        self.P0 = np.diag(prior**2)
        self.C = np.asarray([[measurement_jacobian(k*self.dt, self.reference(k*self.dt), site) @ self.J
                              for site in self.sites] for k in range(self.n)])
        self._observation_cache = {}

    def observation_terms(self, k):
        # Absolute inertial angle advances each orbit; never repeat angle values.
        if k in self._observation_cache:
            return self._observation_cache[k]
        t = k*self.dt
        nominal = np.asarray([angles_at(t, self.reference(t), site) for site in self.sites])
        terms = nominal, self.C[k % self.n].reshape(-1, 6)
        self._observation_cache[k] = terms
        return terms


def _angular_departure(measured, nominal):
    delta = np.asarray(measured)-nominal
    delta[..., 0] = wrap(delta[..., 0])
    return delta.ravel()


def schedule(orbit, visible):
    """Covariances/gains conditioned on actual visibility; no fabricated updates."""
    P = orbit.P0.copy()
    out = []
    for k, mask in enumerate(visible):
        if k:
            A = orbit.A[(k-1) % orbit.n]
            P = A @ P @ A.T + orbit.Q
        _, C = orbit.observation_terms(k)
        selected = np.flatnonzero(np.repeat(mask, 2))
        H = C[selected]
        R = np.eye(len(selected))*orbit.sigma_angle**2
        if len(selected):
            S = H @ P @ H.T + R
            L = np.linalg.cholesky(S)
            K = np.linalg.solve(S, H @ P).T
            M = np.eye(6)-K @ H
            P = M @ P @ M.T + K @ R @ K.T
        else:
            L, K = np.empty((0, 0)), np.empty((6, 0))
        P = (P+P.T)/2
        out.append((selected, H, K, L, P.copy()))
    return out


def predict(orbit, rec, gains, free_run=False, schedules=None):
    """Periodic STM / linearized angular predictor; initial prior is nominal.

    The innovation at an epoch jointly includes all visible LSO angles. The
    Cholesky standardization handles cross-station innovation covariance.
    No direct noisy position/velocity samples enter this calculation.
    """
    schedules = schedules or [schedule(orbit, m) for m in rec.visible]
    priors, navs, errors, standardized, covariances, angle_predictions = [], [], [], [], [], []
    for a, commands in enumerate(rec.u):
        z = np.zeros(6)  # independent nominal prior; not initialized from truth
        prior_rows, nav_rows, eps_rows, white_rows, cov_rows, angle_rows = [], [], [], [], [], []
        for k in range(len(commands)+1):
            if k:
                j = (k-1) % orbit.n
                z = orbit.A[j] @ z + orbit.B[j] @ (np.asarray(gains)*commands[k-1])
            nominal, C = orbit.observation_terms(k)
            selected, H, K, L, P = schedules[a][k]
            prior_rows.append(z*orbit.scale)
            eps = np.full(2*len(orbit.sites), np.nan)
            white = eps.copy()
            predicted_angles = nominal.ravel()+C @ z
            if len(selected):
                e = _angular_departure(rec.angles[a, k], nominal)[selected]-H @ z
                eps[selected] = e
                white[selected] = np.linalg.solve(L, e)
                if not free_run or k == 0:
                    z = z + K @ e
            nav_rows.append(z*orbit.scale)
            eps_rows.append(eps)
            white_rows.append(white)
            cov_rows.append(P*orbit.scale[:, None]*orbit.scale[None, :])
            angle_rows.append(predicted_angles.reshape(-1, 2))
        priors.append(prior_rows); navs.append(nav_rows); errors.append(eps_rows)
        standardized.append(white_rows); covariances.append(cov_rows); angle_predictions.append(angle_rows)
    return dict(predicted=np.asarray(priors), navigation=np.asarray(navs),
                residual=np.asarray(errors), standardized=np.asarray(standardized),
                covariance=np.asarray(covariances), angle_predictions=np.asarray(angle_predictions))


def make_experiment(orbit, count, first_seed):
    U, truth, raw, masks, seeds = [], [], [], [], []
    s = orbit.settings
    sigma0 = np.repeat([s['initial_position_sigma_km'], s['initial_velocity_sigma_m_s']], 3)
    times = np.arange(orbit.n+1)*orbit.dt
    for seed in range(first_seed, first_seed+count):
        rng = np.random.default_rng(seed)
        u = rng.normal(0, s['excitation_sigma_m_s'], (orbit.n, 3))
        state = orbit.state(rng.normal(size=6)*sigma0, 0.)
        states = [state]
        for k, command in enumerate(u):
            state, _, _ = orbit.step(state, k, command, s['simulated_actuator_gains'])
            states.append(state)
        states = np.asarray(states)
        obs = observations(times, states, orbit.cfg, seed+100000)
        U.append(u); seeds.append(seed)
        truth.append([orbit.output(x, t) for x, t in zip(states, times)])
        raw.append(obs['measured_angles_rad']); masks.append(obs['sensor_visible'])
    truth = np.asarray(truth)
    rec = Experiment(np.asarray(U), np.full_like(truth, np.nan), truth, np.array(seeds),
                     np.asarray(raw), np.asarray(masks), np.zeros((*truth.shape[:2], 6, 6)))
    nav = predict(orbit, rec, np.ones(3))
    rec.y, rec.covariance = nav['navigation'], nav['covariance']
    return rec


def prepare_study(root, output):
    cfg, settings = physics.load_settings(root)
    orbit = OrbitModel(cfg, settings)
    data = {name:make_experiment(orbit, settings[name+'_arcs'], settings['seed']+1000*i)
            for i,name in enumerate(('training', 'validation', 'test'))}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, rec in data.items():
        np.savez_compressed(output/(name+'.npz'), **vars(rec))
    meta = dict(configuration=orbit.cfg, settings=settings, measurement_source='lso_angles',
                navigation='Periodic STM filter driven only by visible LSO alpha/delta angles',
                cadence_s=orbit.dt_s, site_count=len(orbit.sites),
                split_seeds={name:rec.seeds.tolist() for name,rec in data.items()},
                sha256={name:hashlib.sha256((output/(name+'.npz')).read_bytes()).hexdigest() for name in data},
                alignment='LSO angles and navigation at k are pre-burn; u[k] affects k+1',
                interpretation='Coasted state estimates remain finite during gaps; raw missing angles remain NaN')
    (output/'experiment.json').write_text(json.dumps(meta, indent=2)+'\n')
    return orbit, data


def restore_study(root, output):
    cfg, settings = physics.load_settings(root)
    output = Path(output)
    if not (output/'experiment.json').exists():
        return prepare_study(root, output)
    meta = json.loads((output/'experiment.json').read_text())
    orbit = OrbitModel(cfg, settings)
    if meta.get('measurement_source') != 'lso_angles' or meta['settings'] != settings or meta['configuration'] != orbit.cfg:
        return prepare_study(root, output)
    data = {}
    for name, digest in meta['sha256'].items():
        p = output/(name+'.npz')
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('LSO experiment checksum mismatch: '+name)
        with np.load(p, allow_pickle=False) as z:
            data[name] = Experiment(**{key:z[key] for key in z.files})
    return orbit, data


def fit_greybox(orbit, training):
    schedules = [schedule(orbit, m) for m in training.visible]
    observed = np.repeat(training.visible, 2, axis=-1)
    if observed.sum() < 30:
        raise ValueError('Insufficient LSO observations for identification')
    def errors(g):
        return predict(orbit, training, g, schedules=schedules)['standardized'][observed]
    result = least_squares(errors, np.ones(3), bounds=(.5, 1.5), x_scale='jac',
                           ftol=1e-9, xtol=1e-9, gtol=1e-8)
    if not result.success:
        raise RuntimeError(result.message)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    covariance = np.linalg.pinv(result.jac.T @ result.jac)
    return dict(gains=result.x, gain_standard_errors=np.sqrt(np.diag(covariance)),
                jacobian_singular_values=singular, cost=float(result.cost),
                optimizer_message=result.message, fitted_parameters=3,
                criterion='visible raw angular innovations; fixed geometry/visibility-conditioned covariance')


def prediction_scores(orbit, rec, gains):
    p = predict(orbit, rec, gains)
    f = predict(orbit, rec, gains, free_run=True)
    def state_score(e):
        return dict(position_rmse_km=float(np.sqrt(np.mean(np.sum(e[..., :3]**2, axis=-1)))),
                    velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum(e[..., 3:]**2, axis=-1)))))
    # Truth is accessed here only for evaluation, never by predict or fit.
    return dict(angular_innovation_rmse_arcsec=float(np.sqrt(np.nanmean(p['residual']**2))/ARCSEC),
                navigation=state_score(p['navigation']-rec.truth),
                free_run=state_score(f['navigation']-rec.truth))


def uncontrolled(orbit):
    s = orbit.settings
    state = orbit.state(np.array([.03,-.02,.01,.0003,-.0002,.0001]), 0.)
    times, outputs = [0.], [orbit.output(state, 0.)]
    reason = 'completed requested propagation'
    for k in range(s['uncontrolled_periods']*orbit.n):
        try:
            state, tt, xx = orbit.step(state, k, np.zeros(3), substeps=4)
        except RuntimeError as exc:
            reason = 'integration stopped: '+str(exc)
            break
        times.extend(tt[1:]); outputs.extend(orbit.output(x,t) for x,t in zip(xx[1:],tt[1:]))
        if np.linalg.norm(outputs[-1][:3]) > s['uncontrolled_stop_km']:
            reason = 'reached declared departure cutoff'
            break
    days = np.asarray(times)*orbit.cfg['time_unit_s']/86400
    return days, np.asarray(outputs), dict(planned_periods=s['uncontrolled_periods'],
                                          last_day=float(days[-1]), stop_reason=reason)


def stationkeeping_trial(orbit, gains, seed=26960, duration_days=None,
                        planning_stride_steps=1, stop_on_departure_cutoff=True):
    """Simulate LSO-guided nonlinear station keeping for a fixed requested span.

    State estimates use visible noisy angles only. The longer annual case can
    reduce planner calls while retaining measurement and state updates at every
    sample. Truth is used only by the sensor simulator and evaluation summary.
    """
    s = orbit.settings
    if duration_days is None:
        count = s['stationkeeping_periods']*orbit.n
        requested_days = count*orbit.dt_s/86400
    else:
        if not np.isfinite(duration_days) or duration_days <= 0:
            raise ValueError('duration_days must be finite and positive')
        requested_days = float(duration_days)
        full_steps = requested_days * 86400 / orbit.dt_s
        nearest_step = round(full_steps)
        count = (nearest_step if nearest_step > 0 and
                 np.isclose(full_steps, nearest_step, rtol=0, atol=1e-11)
                 else int(np.ceil(full_steps)))
    requested_end = requested_days * 86400 / orbit.cfg['time_unit_s']
    ends_on_grid = np.isclose(requested_end, count * orbit.dt, rtol=0, atol=1e-12)
    if planning_stride_steps < 1:
        raise ValueError('planning_stride_steps must be at least 1')
    state = orbit.state(np.array([.03,-.02,.01,.0003,-.0002,.0001]), 0.)
    z, P = np.zeros(6), orbit.P0.copy()
    outputs, days = [orbit.output(state, 0.)], [0.]
    burn_day, commands, phases, nav, raw, visibility, covariance = [], [], [], [], [], [], []
    pending, scheduled, stop_reason = None, None, 'completed requested propagation'
    for k in range(count + int(ends_on_grid)):
        t = k*orbit.dt
        obs = observations(np.array([t]), state[None,:], orbit.cfg, seed+100000+k)
        measured, mask = obs['measured_angles_rad'][0], obs['sensor_visible'][0]
        nominal, C = orbit.observation_terms(k)
        selected = np.flatnonzero(np.repeat(mask,2)); H = C[selected]
        if len(selected):
            R = np.eye(len(selected))*orbit.sigma_angle**2
            innovation = _angular_departure(measured, nominal)[selected]-H @ z
            K = np.linalg.solve(H @ P @ H.T+R, H @ P).T
            z += K @ innovation
            M = np.eye(6)-K @ H
            P = M @ P @ M.T+K @ R @ K.T
        nav.append(z*orbit.scale); raw.append(measured); visibility.append(mask)
        covariance.append(P*orbit.scale[:,None]*orbit.scale[None,:])
        if k == count:
            break
        if stop_on_departure_cutoff and np.linalg.norm(outputs[-1][:3]) > s['uncontrolled_stop_km']:
            stop_reason = 'departure cutoff; remaining span not simulated'
            break
        command = np.zeros(3)
        if pending is None and k % planning_stride_steps == 0:
            plan = physics.plan_burn(orbit, z*orbit.scale, k, gains)
            if plan['delay'] >= 0 and np.linalg.norm(plan['command']) > 1e-10:
                pending, scheduled = plan['command'], k+plan['delay']
        if pending is not None and scheduled == k:
            command, pending = pending, None
            burn_day.append(t*orbit.cfg['time_unit_s']/86400)
            commands.append(command.copy()); phases.append((k % orbit.n)*360/orbit.n)
        end_time = requested_end if k == count - 1 and not ends_on_grid else None
        state, tt, xx = orbit.step(
            state, k, command, s['simulated_actuator_gains'],
            s['nonlinear_samples_per_step'], end_time=end_time,
        )
        outputs.extend(orbit.output(x,epoch) for x,epoch in zip(xx,tt))
        days.extend(tt*orbit.cfg['time_unit_s']/86400)
        A, B = orbit.A[k % orbit.n], orbit.B[k % orbit.n]
        z = A @ z+B @ (np.asarray(gains)*command)
        P = A @ P @ A.T+orbit.Q
        P = (P+P.T)/2
    y, day = np.asarray(outputs), np.asarray(days)
    pos, vel = np.linalg.norm(y[:,:3],axis=1),np.linalg.norm(y[:,3:],axis=1)
    outside = (pos>s['position_limit_km']) | (vel>s['velocity_limit_m_s'])
    total_delta_v=float(sum(np.linalg.norm(u) for u in commands))
    completed = bool(np.isclose(day[-1], requested_days, rtol=0, atol=1e-9))
    annual_horizon = s.get('annual_horizon_days')
    annual_requested = (annual_horizon is not None and
                        np.isclose(requested_days, annual_horizon, rtol=0, atol=1e-9))
    summary = dict(seed=seed, measurement_source='lso_angles',
                   requested_duration_days=requested_days,
                   planning_stride_steps=int(planning_stride_steps),
                   completed=completed, stop_reason=stop_reason, last_day=float(day[-1]),
                   maximum_position_km=float(pos.max()), maximum_velocity_m_s=float(vel.max()),
                   all_evaluated_samples_inside=bool(not outside.any()),
                   first_sampled_exit_day=float(day[outside][0]) if outside.any() else None,
                   burns=len(commands), total_commanded_delta_v_m_s=total_delta_v,
                   annual_delta_v_m_s=total_delta_v if annual_requested and completed else None,
                   visible_epoch_fraction=float(np.mean(np.any(visibility,axis=1))),
                   interpretation='LSO-derived guidance; truth only for sensor simulation/evaluation; sampled corridor test')
    return dict(day=day, output=y, burn_day=np.asarray(burn_day),commands=np.asarray(commands).reshape(-1,3),
                phase_deg=np.asarray(phases),navigation=np.asarray(nav),angles=np.asarray(raw),
                visible=np.asarray(visibility),covariance=np.asarray(covariance),summary=summary)
