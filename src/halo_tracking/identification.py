"""Generate labeled, phase-aware CR3BP station-keeping impulse examples."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import rhs, jacobian, propagate, propagate_step
from .estimation import initial_covariance
from .frames import rotation, state_transform
from .measurements import predict, measurement_jacobian, wrap, ARCSEC
from .network import observations, sensor_configs
from .orbit_reference import correct_halo_reference


def physical_deviation(time_tu, state, nominal, cfg):
    """Return inertial position difference in km and velocity in m/s."""
    delta = state_transform(time_tu, cfg) @ (state-nominal)
    return np.r_[delta[:3], 1000*delta[3:]]


def _navigation(times, measured, masks, truth_initial, cfg, seed):
    """Propagate one observer EKF and assimilate each accepted LSO angle pair."""
    P = initial_covariance(cfg)
    rng = np.random.default_rng(seed)
    x = truth_initial + rng.multivariate_normal(np.zeros(6), P)
    R = np.eye(2)*(cfg['measurement']['angle_sigma_arcsec']*ARCSEC)**2
    sites = sensor_configs(cfg)
    states, covariance, age = [], [], []
    previous = times[0]
    last = None
    for k, t in enumerate(times):
        x, F, Q = propagate_step(
            x, previous, t, cfg['mu'],
            cfg['estimation']['acceleration_psd_normalized'])
        P = F@P@F.T+Q
        for j, site in enumerate(sites):
            if not masks[k, j]:
                continue
            H = measurement_jacobian(t, x, site, False)
            innovation = measured[k, j] - predict(t, x, site, False)
            innovation[0] = wrap(innovation[0])
            S = H@P@H.T+R
            K = np.linalg.solve(S, H@P).T
            x = x + K@innovation
            I_KH = np.eye(6)-K@H
            P = I_KH@P@I_KH.T+K@R@K.T
            P = (P+P.T)/2
            last = t
        states.append(x.copy())
        covariance.append(np.diag(P).copy())
        age.append(np.nan if last is None else (t-last)*cfg['time_unit_s']/86400.)
        previous = t
    return np.array(states), np.array(covariance), np.array(age)


def _forward(state, start, duration, cfg, stm):
    """Propagate the CR3BP state and optionally its 6-by-6 STM."""
    end = start+duration
    if stm:
        def augmented(t, y):
            return np.r_[rhs(t, y[:6], cfg['mu']),
                         (jacobian(y[:6], cfg['mu'])@
                          y[6:].reshape(6, 6)).ravel()]
        initial = np.r_[state, np.eye(6).ravel()]
        sol = solve_ivp(augmented, (start, end), initial,
                        method='DOP853', rtol=2e-10, atol=2e-12)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)
    sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']),
                    (start, end), state,
                    method='DOP853', rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1], None


def _impulse_label(time_tu, state_estimate, nominal_at, cfg, policy):
    """Solve a weighted linearized single-impulse terminal correction.

    A burn modifies velocity only, not position. The target is the corrected
    nominal state a short lead time later. Compute the local STM at the
    estimated state, regularize the weighted terminal errors, then verify the
    nonlinear corrected propagation; infeasible maneuvers are MASKED rather
    than silently clipping to the spacecraft burn budget.
    """
    L, TU = cfg['length_unit_km'], cfg['time_unit_s']
    lead = policy['guidance_horizon_fraction']*policy['period_tu']
    terminal_nominal = nominal_at(time_tu+lead)
    nominal_now = nominal_at(time_tu)
    deviation = physical_deviation(time_tu, state_estimate, nominal_now, cfg)
    dr, dv = np.linalg.norm(deviation[:3]), np.linalg.norm(deviation[3:])
    trigger = (dr >= policy['trigger_position_km'] or
               dv >= policy['trigger_velocity_m_s'])
    before, stm = _forward(state_estimate, time_tu, lead, cfg, bool(trigger))
    B = state_transform(time_tu+lead, cfg)
    physical_free = B@(before-terminal_nominal)
    endpoint_before = np.r_[physical_free[:3], 1000*physical_free[3:]]
    weight = np.array([1/policy['limit_position_km']]*3 +
                      [1/policy['limit_velocity_m_s']]*3)
    baseline_cost = np.linalg.norm(weight*endpoint_before)
    burn = np.zeros(3)
    feasible = True
    predicted_after = baseline_cost
    if trigger:
        # Convert a 1 m/s velocity impulse into normalized synodic velocity.
        velocity_to_normalized = TU/(1000*L)
        impulse_to_terminal = stm[:, 3:]*velocity_to_normalized
        physical_sensitivity = B@impulse_to_terminal
        physical_sensitivity[3:] *= 1000
        A = weight[:, None]*physical_sensitivity
        b = weight*endpoint_before
        regularizer = policy['regularization']/policy['maximum_impulse_m_s']
        A_aug = np.vstack((A, regularizer*np.eye(3)))
        b_aug = np.r_[-b, np.zeros(3)]
        candidate = np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]
        feasible = (np.all(np.isfinite(candidate)) and
                    np.linalg.norm(candidate) <= policy['maximum_impulse_m_s'])
        if feasible:
            proposed = state_estimate.copy()
            proposed[3:] += candidate*velocity_to_normalized
            after, _ = _forward(proposed, time_tu, lead, cfg, False)
            physical_after = B@(after-terminal_nominal)
            endpoint_after = np.r_[physical_after[:3],
                                   1000*physical_after[3:]]
            predicted_after = np.linalg.norm(weight*endpoint_after)
            feasible = (np.isfinite(predicted_after) and
                        predicted_after < baseline_cost)
        if feasible:
            burn = candidate
        else:
            burn[:] = np.nan
            predicted_after = np.nan
    return dict(trigger=bool(trigger), feasible=bool(feasible),
                burn_rotating_m_s=burn,
                burn_mci_m_s=rotation(time_tu)@burn,
                position_deviation_km=dr, velocity_deviation_m_s=dv,
                predicted_free_weighted_error=baseline_cost,
                predicted_controlled_weighted_error=predicted_after)


def generate_dataset(cfg, directory):
    """Generate independent guidance episodes, EKF inputs and checked Δv labels."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(cfg)
    setting = cfg['identification']
    nominal_cfg = correct_halo_reference(cfg)
    T = float(nominal_cfg['orbit']['period_tu'])
    nominal_seed = np.asarray(nominal_cfg['orbit']['state'], dtype=float)
    rounded_seed = np.asarray(cfg['orbit']['state'], dtype=float)
    arc_periods = float(setting['arc_duration_periods'])
    decisions_per_period = int(setting['decision_epochs_per_period'])
    dt_sensor = float(setting['measurement_cadence_s'])
    if (arc_periods <= 0 or decisions_per_period < 1 or dt_sensor <= 0):
        raise ValueError('Invalid guidance sampling settings')
    for name in ('trigger_position_km', 'trigger_velocity_m_s',
                 'limit_position_km', 'limit_velocity_m_s',
                 'maximum_impulse_m_s'):
        if setting[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if (setting['trigger_position_km'] >= setting['limit_position_km']
            or setting['trigger_velocity_m_s'] >= setting['limit_velocity_m_s']):
        raise ValueError('Guidance trigger thresholds must precede corridor limits')
    horizon = arc_periods*T
    phases = np.linspace(0, horizon,
                         int(round(arc_periods*decisions_per_period))+1)
    observation_times = np.arange(0, horizon, dt_sensor/cfg['time_unit_s'])
    times = np.unique(np.r_[observation_times, phases, horizon])
    indices = np.searchsorted(times, phases)
    if not np.allclose(times[indices], phases, atol=1e-12):
        raise RuntimeError('Guidance epochs are missing from observation grid')
    sensor_cfg = deepcopy(cfg)
    sensor_cfg['simulation']['cadence_s'] = dt_sensor

    reference_sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']), (0, T),
                              nominal_seed, method='DOP853',
                              dense_output=True, rtol=2e-11, atol=2e-13)
    if not reference_sol.success:
        raise RuntimeError(reference_sol.message)
    def nominal_at(time_tu):
        return reference_sol.sol(np.mod(time_tu, T))

    policy = {name: setting[name] for name in (
        'trigger_position_km', 'trigger_velocity_m_s',
        'limit_position_km', 'limit_velocity_m_s',
        'maximum_impulse_m_s', 'guidance_horizon_fraction',
        'regularization')}
    policy['period_tu'] = T
    spread = np.array(
        [setting['position_spread_km']/cfg['length_unit_km']]*3 +
        [setting['velocity_spread_m_s']*.001*cfg['time_unit_s']/
         cfg['length_unit_km']]*3)
    streams = np.random.SeedSequence(setting['seed']).spawn(3)
    summary = {}

    for split, stream in zip(('training', 'validation', 'test'), streams):
        rng = np.random.default_rng(stream)
        n = int(setting[split+'_arcs'])
        if n < 1:
            raise ValueError('Each data split must contain at least one arc')
        all_data = []
        initial_states, source_type, magnitude_scale = [], [], []
        for arc in range(n):
            # Mix actual rounded-catalog departure with nearby corrected
            # nominal perturbations; each split has its own independent RNG.
            from_catalog = (arc % 2 == 0)
            scale = float(rng.choice([1., 5., 20.]))
            x0 = ((rounded_seed if from_catalog else nominal_seed)
                  + rng.normal(size=6)*spread*scale)
            x = propagate(x0, times, cfg['mu'])
            obs = observations(times, x, sensor_cfg,
                               int(rng.integers(0, 2**32)))
            nav, cov, age = _navigation(
                times, obs['measured_angles_rad'], obs['sensor_visible'],
                x0, sensor_cfg, int(rng.integers(0, 2**32)))
            decisions = nav[indices]
            nominals = np.asarray([nominal_at(t) for t in phases])
            labels = [_impulse_label(t, state, nominal_at, cfg, policy)
                      for t, state in zip(phases, decisions)]
            all_data.append(dict(
                truth_rotating_normalized=x,
                navigation_rotating_normalized=nav,
                navigation_covariance_diagonal_normalized=cov,
                navigation_lso_observation_age_days=age,
                measured_angles_rad=obs['measured_angles_rad'],
                sensor_visible=obs['sensor_visible'],
                sensor_elevation_rad=obs['sensor_elevation_rad'],
                decision_truth_states_normalized=x[indices],
                decision_navigation_states_normalized=decisions,
                decision_nominal_states_normalized=nominals,
                decision_navigation_covariance_diagonal_normalized=cov[indices],
                decision_lso_observation_age_days=age[indices],
                decision_observation_counts=np.sum(
                    obs['sensor_visible'][indices], axis=1),
                decision_navigation_deviation_mci_km_m_s=np.asarray([
                    physical_deviation(t, state, nominal, cfg)
                    for t, state, nominal in zip(phases, decisions, nominals)]),
                maneuver_required=np.asarray(
                    [label['trigger'] for label in labels]),
                maneuver_label_valid=np.asarray(
                    [label['feasible'] for label in labels]),
                target_delta_v_rotating_m_s=np.asarray(
                    [label['burn_rotating_m_s'] for label in labels]),
                target_delta_v_mci_m_s=np.asarray(
                    [label['burn_mci_m_s'] for label in labels]),
                predicted_no_burn_weighted_error=np.asarray(
                    [label['predicted_free_weighted_error'] for label in labels]),
                predicted_with_burn_weighted_error=np.asarray(
                    [label['predicted_controlled_weighted_error'] for label in labels])))
            initial_states.append(x0)
            source_type.append(int(from_catalog))
            magnitude_scale.append(scale)
        fields = list(all_data[0])
        arrays = {field: np.stack([a[field] for a in all_data])
                  for field in fields}
        arrays.update(
            times_tu=times, times_s=times*cfg['time_unit_s'],
            decision_times_tu=phases,
            decision_times_s=phases*cfg['time_unit_s'],
            decision_phase_rad=2*np.pi*np.mod(phases, T)/T,
            initial_states_normalized=np.asarray(initial_states),
            initial_source_catalog=np.asarray(source_type, dtype=bool),
            initial_perturbation_scale=np.asarray(magnitude_scale))
        np.savez_compressed(directory/(split+'.npz'), **arrays)
        trigger = arrays['maneuver_required']
        valid = arrays['maneuver_label_valid']
        summary[split] = dict(
            arcs=n, samples_per_arc=len(times),
            decisions_per_arc=len(phases), decisions=n*len(phases),
            triggered=int(np.sum(trigger)),
            valid_triggered=int(np.sum(trigger & valid)),
            rejected=int(np.sum(trigger & ~valid)),
            no_burn=int(np.sum(~trigger)),
            visible_fraction=float(np.mean(
                np.any(arrays['sensor_visible'], axis=2))))
    metadata = dict(
        purpose='Supervised station-keeping impulse prediction, not next-state dynamics identification.',
        configuration=cfg,
        nominal_reference=dict(period_tu=T, corrected_initial_state=nominal_seed.tolist(),
                               closure=nominal_cfg['orbit']['correction']),
        guidance_policy=policy,
        data_axes='trajectories: arc,time,...; decisions: arc,decision,...',
        inputs=('Use decision_navigation_states_normalized, decision_nominal_states_normalized, '
                'decision_navigation_deviation_mci_km_m_s, covariance, phase and observation age. '
                'Do not use decision_truth_states_normalized as a deployable feature.'),
        labels=('target_delta_v_rotating_m_s is a three-axis impulsive velocity change, '
                'target_delta_v_mci_m_s is that vector rotated into MCI, not a velocity error. '
                'maneuver_required reflects estimated trigger thresholds; no-trigger labels are zero. '
                'Rejected/unfeasible labels are NaN and must be masked by maneuver_label_valid.'),
        method=('Local CR3BP STM with weighted terminal position and velocity residual, '
                'regularized minimum-effort velocity-only impulse; nonlinear forward check '
                'must improve weighted endpoint residual. Single maneuver, fixed '
                'guidance horizon, no multi-burn optimal-control or actuator model.'),
        limits=('100 km / 1 m/s corridor, triggered at 50 km / 0.5 m/s, '
                'chosen from rounded seed vs corrected nominal: ~23 km / 0.13 m/s at t=0, '
                '~110 km / 0.9 m/s by 0.5T, >500 km by 1.5T. '
                'Exploratory control-design values, not flight requirements.'),
        warnings=('Target Δv is computed from EKF navigation estimates, not hidden true states. '
                  'True state is saved for evaluation only. Nominal periodic-reference '
                  'uncertainty and navigation filter consistency require further study. '
                  'Invalid requested impulses are not silently capped.'),
        splits='Independent simulated arcs with split-specific random seeds; never mix time samples across splits.',
        summary=summary)
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return summary
