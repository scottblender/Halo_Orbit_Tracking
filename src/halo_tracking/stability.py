"""Local Floquet diagnostics and finite-horizon nominal-orbit sensitivity."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import rhs, jacobian
from .frames import state_transform
from .orbit_reference import correct_halo_reference


def monodromy(state, period_tu, mu, rtol=2e-12, atol=2e-14):
    """Integrate a state and identity STM for one period; return endpoint, STM."""
    if not np.isfinite(period_tu) or period_tu <= 0:
        raise ValueError('Period must be positive and finite')
    def augmented(t, y):
        return np.r_[rhs(t, y[:6], mu),
                     (jacobian(y[:6], mu) @ y[6:].reshape(6, 6)).ravel()]
    sol = solve_ivp(augmented, (0., period_tu),
                    np.r_[state, np.eye(6).ravel()], method='DOP853',
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)


def orbit_stability(cfg, periods=5, ensemble_count=8, samples_per_period=100,
                    position_spread_km=1., velocity_spread_m_s=.01,
                    position_limit_km=100., velocity_limit_m_s=1., seed=921):
    """Analyze corrected nominal stability, independently of catalog departure.

    Two integrations check spectral-radius sensitivity to solver tolerances.
    Near-unit multipliers are reported as unresolved, never proof of nonlinear
    stability. Ensembles start around the corrected reference, use no burns,
    and report sampled, equal-phase physical departure over a finite horizon.
    """
    for name, value, minimum in [('periods', periods, 1),
                                  ('ensemble_count', ensemble_count, 1),
                                  ('samples_per_period', samples_per_period, 10)]:
        if not np.isfinite(value) or value < minimum or int(value) != value:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    values = (position_spread_km, velocity_spread_m_s,
              position_limit_km, velocity_limit_m_s)
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('Spreads and corridor limits must be positive and finite')
    periods, ensemble_count, samples_per_period = map(
        int, (periods, ensemble_count, samples_per_period))
    nominal = correct_halo_reference(cfg)
    x0 = np.asarray(nominal['orbit']['state'])
    T, mu = nominal['orbit']['period_tu'], cfg['mu']
    _, coarse = monodromy(x0, T, mu, rtol=2e-10, atol=2e-12)
    endpoint, matrix = monodromy(x0, T, mu)
    multipliers = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(multipliers)))
    coarse_radius = float(np.max(np.abs(np.linalg.eigvals(coarse))))
    change = abs(radius-coarse_radius)
    margin = max(1e-4, 10*change)
    closure = state_transform(T, cfg) @ (endpoint-x0)
    closure_ok = (np.linalg.norm(closure[:3]) < .001 and
                  1000*np.linalg.norm(closure[3:]) < .001)
    unstable = closure_ok and radius > 1+margin
    classification = ('linearly unstable' if unstable else
                      'unresolved: no robust instability detected')
    T_days = T*cfg['time_unit_s']/86400
    efold = float(T_days/np.log(radius)) if unstable else None

    times = np.linspace(0., periods*T, periods*samples_per_period+1)
    def integrate(initial, end, **kwargs):
        sol = solve_ivp(lambda t, x: rhs(t, x, mu), (0., end), initial,
                        method='DOP853', rtol=2e-12, atol=2e-14, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol
    reference = integrate(x0, T, dense_output=True)
    phase = (np.arange(len(times)) % samples_per_period)*T/samples_per_period
    repeated = reference.sol(phase).T
    scale = np.r_[[position_spread_km/cfg['length_unit_km']]*3,
                  [velocity_spread_m_s*cfg['time_unit_s']/
                   (1000*cfg['length_unit_km'])]*3]
    offsets = np.random.default_rng(seed).normal(size=(ensemble_count, 6))*scale
    positions, velocities = [], []
    for offset in offsets:
        states = integrate(x0+offset, times[-1], t_eval=times).y.T
        deviation = np.array([state_transform(t, cfg) @ dx
                              for t, dx in zip(times, states-repeated)])
        positions.append(np.linalg.norm(deviation[:, :3], axis=1))
        velocities.append(1000*np.linalg.norm(deviation[:, 3:], axis=1))
    positions, velocities = np.array(positions), np.array(velocities)
    outside = (positions >= position_limit_km) | (velocities >= velocity_limit_m_s)
    first_crossing = [float(times[np.flatnonzero(row)[0]]*cfg['time_unit_s']/86400)
                      if row.any() else None for row in outside]
    summary = dict(
        classification=classification, nominal_period_days=float(T_days),
        mu=float(mu), length_unit_km=cfg['length_unit_km'],
        time_unit_s=cfg['time_unit_s'],
        corrected_state_normalized=x0.tolist(), period_tu=float(T),
        position_closure_km=float(np.linalg.norm(closure[:3])),
        velocity_closure_m_s=float(1000*np.linalg.norm(closure[3:])),
        floquet_multipliers=[dict(real=float(v.real), imag=float(v.imag),
                                  magnitude=float(abs(v))) for v in multipliers],
        spectral_radius=radius, coarse_spectral_radius=coarse_radius,
        spectral_radius_tolerance_change=change, classification_margin=margin,
        determinant=float(np.linalg.det(matrix)), efolding_time_days=efold,
        periods=periods, ensemble_count=ensemble_count,
        samples_per_period=samples_per_period, seed=int(seed),
        position_spread_km_per_axis=position_spread_km,
        velocity_spread_m_s_per_axis=velocity_spread_m_s,
        position_limit_km=position_limit_km, velocity_limit_m_s=velocity_limit_m_s,
        arcs_with_sampled_corridor_exit=int(np.sum(np.any(outside, axis=1))),
        first_sampled_exit_days=first_crossing,
        interpretation=('Floquet results are local to the corrected periodic CR3BP orbit. '
                        'E-folding is asymptotic growth along the dominant mode, not '
                        'a tracking deadline or maneuver budget. Tolerance comparison '
                        'is a numerical sensitivity check, not a rigorous error bound. '
                        'Finite-horizon ensemble non-exits are right-censored; '
                        'neither they nor near-unit multipliers prove nonlinear stability.'))
    return dict(summary=summary, monodromy=matrix, times_tu=times,
                initial_offsets_normalized=offsets,
                position_departure_km=positions, velocity_departure_m_s=velocities)


def save_stability(result, directory):
    """Save JSON diagnostics and reproducible ensemble histories."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'stability_summary.json').write_text(
        json.dumps(result['summary'], indent=2, allow_nan=False)+'\n')
    np.savez_compressed(directory/'stability_histories.npz',
                        **{k: v for k, v in result.items() if k != 'summary'})
