"""Uncontrolled catalog-seed departure from a corrected periodic CR3BP reference."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import rhs, jacobi
from .frames import state_transform
from .orbit_reference import correct_halo_reference


def reference_departure(cfg, periods=20, samples_per_period=500,
                        rtol=2e-11, atol=2e-13, ensemble_count=0):
    """Integrate the original seed continuously against a periodic nominal arc.

    The *truth* initial state remains the supplied rounded catalog state.
    Only the reference initial state and period undergo differential correction.
    """
    if (periods < 2 or int(periods) != periods
            or samples_per_period < 10 or int(samples_per_period) != samples_per_period):
        raise ValueError('Use integer periods >=2 and at least 10 samples per period')
    if ensemble_count < 0 or int(ensemble_count) != ensemble_count:
        raise ValueError('ensemble_count must be a nonnegative integer')
    periods, samples_per_period = int(periods), int(samples_per_period)
    corrected_cfg = correct_halo_reference(cfg)
    period = corrected_cfg['orbit']['period_tu']
    x0 = np.asarray(cfg['orbit']['state'], dtype=float)
    nominal0 = np.asarray(corrected_cfg['orbit']['state'], dtype=float)

    def integrate(state, end):
        sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']), (0, end), state,
                        method='DOP853', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    nominal = integrate(nominal0, period)
    trajectory = integrate(x0, periods*period)
    times = np.linspace(0, periods*period, periods*samples_per_period+1)
    phase = (np.arange(len(times)) % samples_per_period)*period/samples_per_period
    states = trajectory.sol(times).T
    repeated = nominal.sol(phase).T
    reference_period_states = nominal.sol(
        np.linspace(0, period, samples_per_period+1)).T
    delta = states-repeated
    physical = np.array([state_transform(t, cfg) @ dx
                         for t, dx in zip(times, delta)])
    position = np.linalg.norm(physical[:, :3], axis=1)
    velocity = 1000*np.linalg.norm(physical[:, 3:], axis=1)
    prms = np.sqrt(np.mean(position[:-1].reshape(periods, samples_per_period)**2, axis=1))
    vrms = np.sqrt(np.mean(velocity[:-1].reshape(periods, samples_per_period)**2, axis=1))
    closure = state_transform(period, cfg) @ (
        reference_period_states[-1]-nominal0)

    ensemble_initial = np.empty((int(ensemble_count), 6))
    ensemble_pos = np.empty((int(ensemble_count), len(times)))
    ensemble_vel = np.empty_like(ensemble_pos)
    if ensemble_count:
        L, TU = cfg['length_unit_km'], cfg['time_unit_s']
        spread = np.array([1./L]*3+[0.01*0.001*TU/L]*3)
        rng = np.random.default_rng(cfg['simulation']['seed']+921)
        for i in range(int(ensemble_count)):
            ensemble_initial[i] = rng.normal(size=6)*spread
            perturbed = integrate(x0+ensemble_initial[i], periods*period)
            difference = perturbed.sol(times).T-repeated
            ensemble_pos[i] = L*np.linalg.norm(difference[:, :3], axis=1)
            cross = np.column_stack((-difference[:, 1], difference[:, 0],
                                     np.zeros(len(times))))
            ensemble_vel[i] = (1000*L/TU *
                               np.linalg.norm(difference[:, 3:]+cross, axis=1))

    diagnostics = dict(
        position_closure_km=float(np.linalg.norm(closure[:3])),
        velocity_closure_m_s=float(1000*np.linalg.norm(closure[3:])),
        catalog_closure_km=float(corrected_cfg['orbit']['correction'].get(
            'original_position_closure_km', np.nan)),
        initial_nominal_offset_km=float(
            np.linalg.norm((x0-nominal0)[:3])*cfg['length_unit_km']),
        nominal_period_tu=float(period),
        catalog_period_tu=float(cfg['orbit']['period_tu']),
        jacobi_range=float(np.ptp([jacobi(x, cfg['mu']) for x in states])),
        periods=periods, samples_per_period=samples_per_period,
        ensemble_count=int(ensemble_count),
        ensemble_position_spread_km_per_axis=1.,
        ensemble_velocity_spread_m_s_per_axis=.01,
        rtol=rtol, atol=atol,
        interpretation=('Original rounded observer state is propagated continuously. '
                        'Only the nominal reference is differentially corrected; '
                        'no station keeping or physical state resets occur.'))

    return dict(times=times, states=states,
                repeated_reference=repeated,
                reference_period_states=reference_period_states,
                position_departure_km=position,
                velocity_departure_m_s=velocity,
                ensemble_initial_perturbations_normalized=ensemble_initial,
                ensemble_position_departure_km=ensemble_pos,
                ensemble_velocity_departure_m_s=ensemble_vel,
                period_position_rms_km=prms,
                period_velocity_rms_m_s=vrms, diagnostics=diagnostics)


def save_departure(result, cfg, directory):
    """Save continuous histories, per-period summaries and nominal closure."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'departure.npz',
                        **{k: v for k, v in result.items() if k != 'diagnostics'})
    table = np.c_[np.arange(1, len(result['period_position_rms_km'])+1),
                  result['period_position_rms_km'],
                  result['period_velocity_rms_m_s']]
    np.savetxt(directory/'period_departure.csv', table, delimiter=',',
               header='period,position_rms_km,velocity_rms_m_s', comments='')
    (directory/'summary.json').write_text(json.dumps(result['diagnostics'], indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
