"""Uncontrolled departure from a repeated one-period CR3BP reference."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import rhs, jacobi
from .frames import state_transform


def reference_departure(cfg, periods=20, samples_per_period=500, rtol=2e-11, atol=2e-13,
                        ensemble_count=0):
    """Compare equal-phase states; keep reference resets explicit."""
    if periods < 2 or int(periods) != periods or samples_per_period < 10:
        raise ValueError('Use an integer number of periods >=2 and at least 10 samples per period')
    if ensemble_count < 0 or int(ensemble_count) != ensemble_count:
        raise ValueError('ensemble_count must be a nonnegative integer')
    period = cfg['orbit']['period_tu']
    x0 = np.asarray(cfg['orbit']['state'])
    def integrate(end):
        sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']), (0, end), x0,
                        method='DOP853', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol
    reference = integrate(period)
    trajectory = integrate(periods*period)
    times = np.linspace(0, periods*period, periods*samples_per_period+1)
    phase_index = np.arange(len(times)) % samples_per_period
    phase = phase_index*period/samples_per_period
    states = trajectory.sol(times).T
    repeated = reference.sol(phase).T
    delta = states-repeated
    physical = np.array([state_transform(t, cfg)@dx for t, dx in zip(times, delta)])
    position = np.linalg.norm(physical[:, :3], axis=1)
    velocity = 1000*np.linalg.norm(physical[:, 3:], axis=1)
    # Exclude the right boundary: each period has the same phase samples.
    prms = np.sqrt(np.mean(position[:-1].reshape(periods, samples_per_period)**2, axis=1))
    vrms = np.sqrt(np.mean(velocity[:-1].reshape(periods, samples_per_period)**2, axis=1))
    closure = state_transform(period, cfg)@(reference.sol(period)-x0)
    perturbations = np.empty((int(ensemble_count), 6))
    ensemble_position_km = np.empty((int(ensemble_count), len(times)))
    ensemble_velocity_m_s = np.empty_like(ensemble_position_km)
    if ensemble_count:
        L, T = cfg['length_unit_km'], cfg['time_unit_s']
        spread = np.array([1./L]*3 + [0.01*.001*T/L]*3)
        rng = np.random.default_rng(cfg['simulation']['seed'] + 921)
        for i in range(int(ensemble_count)):
            perturbations[i] = rng.normal(size=6)*spread
            perturbed0 = x0 + perturbations[i]
            sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']),
                            (0, periods*period), perturbed0, method='DOP853',
                            dense_output=True, rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(f'Perturbed trajectory {i}: {sol.message}')
            difference = sol.sol(times).T - repeated
            ensemble_position_km[i] = L*np.linalg.norm(difference[:, :3], axis=1)
            # Physical inertial velocity includes the rotating-frame cross term.
            cross = np.column_stack((-difference[:, 1], difference[:, 0],
                                     np.zeros(len(times))))
            ensemble_velocity_m_s[i] = (1000*L/T*
                np.linalg.norm(difference[:, 3:] + cross, axis=1))
    diagnostic = dict(position_closure_km=float(np.linalg.norm(closure[:3])),
        velocity_closure_m_s=float(1000*np.linalg.norm(closure[3:])),
        jacobi_range=float(np.ptp([jacobi(x, cfg['mu']) for x in states])),
        periods=int(periods), samples_per_period=int(samples_per_period),
        ensemble_count=int(ensemble_count),
        ensemble_position_spread_km_per_axis=1.,
        ensemble_velocity_spread_m_s_per_axis=0.01,
        rtol=rtol, atol=atol,
        interpretation='Departure from a repeated rounded reference, not a chaos or stability test. Period RMS includes accumulated phase drift.')
    return dict(times=times, states=states, repeated_reference=repeated,
        position_departure_km=position, velocity_departure_m_s=velocity,
        ensemble_initial_perturbations_normalized=perturbations,
        ensemble_position_departure_km=ensemble_position_km,
        ensemble_velocity_departure_m_s=ensemble_velocity_m_s,
        period_position_rms_km=prms, period_velocity_rms_m_s=vrms,
        diagnostics=diagnostic)


def save_departure(result, cfg, directory):
    """Save histories, period summaries and reference closure."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'departure.npz', **{k:v for k,v in result.items() if k!='diagnostics'})
    table = np.c_[np.arange(1, len(result['period_position_rms_km'])+1),
                  result['period_position_rms_km'], result['period_velocity_rms_m_s']]
    np.savetxt(directory/'period_departure.csv', table, delimiter=',',
               header='period,position_rms_km,velocity_rms_m_s', comments='')
    (directory/'summary.json').write_text(json.dumps(result['diagnostics'], indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
