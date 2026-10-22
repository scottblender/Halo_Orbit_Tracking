"""Differential correction of the rounded southern NRHO CR3BP reference."""
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .dynamics import rhs, jacobian


def correct_halo_reference(cfg, tolerance_km=1e-3):
    """Shoot a nearby periodic solution with an autonomous phase constraint.

    Six periodic boundary conditions and a phase condition are solved for the
    six initial components plus the period. This is a local correction of
    the catalog's rounded state, not an independent orbit-family search.
    """
    output = deepcopy(cfg)
    original = np.asarray(output['orbit']['state'], dtype=float)
    period = float(output['orbit']['period_tu'])
    if output['orbit'].get('reference_corrected'):
        return output
    mu = output['mu']
    phase_direction = rhs(0., original, mu)
    cache = {}

    def integrate(z):
        key = tuple(z)
        if key in cache:
            return cache[key]
        x0, T = z[:6], z[6]
        if T <= 0:
            raise ValueError('Reference period must remain positive')

        def augmented(t, y):
            A = jacobian(y[:6], mu)
            return np.r_[rhs(t, y[:6], mu), (A @ y[6:].reshape(6, 6)).ravel()]

        sol = solve_ivp(augmented, (0., T), np.r_[x0, np.eye(6).ravel()],
                        method='DOP853', rtol=4e-12, atol=4e-14)
        if not sol.success:
            raise RuntimeError(sol.message)
        terminal = sol.y[:6, -1]
        stm = sol.y[6:, -1].reshape(6, 6)
        cache.clear()
        cache[key] = (terminal, stm)
        return terminal, stm

    def residual(z):
        terminal, _ = integrate(z)
        return np.r_[terminal - z[:6],
                     phase_direction @ (z[:6] - original)]

    def derivative(z):
        terminal, stm = integrate(z)
        return np.block([[stm - np.eye(6), rhs(0., terminal, mu)[:, None]],
                         [phase_direction[None, :], np.zeros((1, 1))]])

    initial = np.r_[original, period]
    solution = least_squares(residual, initial, jac=derivative, method='lm',
                             max_nfev=80, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    closing = residual(solution.x)[:6]
    position_km = np.linalg.norm(closing[:3]) * output['length_unit_km']
    velocity_m_s = (1000 * np.linalg.norm(closing[3:])
                    * output['length_unit_km'] / output['time_unit_s'])
    if not solution.success or position_km > tolerance_km or velocity_m_s > 1e-3:
        raise RuntimeError('Periodic reference correction failed to close: '
                           f'{position_km:g} km, {velocity_m_s:g} m/s')

    state = solution.x[:6]
    if np.linalg.norm((state-original)[:3]) * output['length_unit_km'] > 1000:
        raise RuntimeError('Correction moved too far from catalog initial state')
    output['orbit'].update(
        state=state.tolist(), period_tu=float(solution.x[6]),
        reference_corrected=True,
        source_precision='Numerically differentially corrected from rounded SNRHL2 seed',
        correction=dict(original_state=original.tolist(), original_period_tu=period,
                        initial_position_change_km=float(
                            np.linalg.norm((state-original)[:3]) * output['length_unit_km']),
                        initial_velocity_change_m_s=float(
                            1000*np.linalg.norm((state-original)[3:]) *
                            output['length_unit_km']/output['time_unit_s']),
                        position_closure_km=float(position_km),
                        velocity_closure_m_s=float(velocity_m_s)))
    return output
