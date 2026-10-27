"""Strict CR3BP periodicity verification for Gateway's supplied 9:2 NRHO seed.

The seed is not arbitrarily propagated for a year: evaluate a numerically
verified one-period solution at the appropriate phase of each time epoch.
This establishes periodic closure in the chosen CR3BP, not ephemeris fidelity.
"""
import numpy as np
from .dynamics import propagate
from .narx_dataset import _reference


def gateway_periodic_reference(cfg, position_tolerance_km=0.001,
                               velocity_tolerance_m_s=0.001):
    state = np.asarray(cfg["gateway"]["state"], dtype=float)
    period = float(cfg["gateway"]["period_tu"])
    if state.shape != (6,) or not np.isfinite(state).all() or period <= 0:
        raise ValueError("Invalid Gateway periodic seed")
    physical_days = period * cfg["time_unit_s"] / 86400
    if not 6.0 < physical_days < 7.0:
        raise ValueError("Gateway seed period is not consistent with the ~6.5-day 9:2 NRHO")
    closure = propagate(state, np.array([0., period]), cfg["mu"])[-1] - state
    position_error = float(np.linalg.norm(closure[:3]) * cfg["length_unit_km"])
    velocity_error = float(np.linalg.norm(closure[3:]) * cfg["length_unit_km"]
                           / cfg["time_unit_s"] * 1000.)
    if position_error > position_tolerance_km or velocity_error > velocity_tolerance_m_s:
        raise ValueError(
            f"Gateway CR3BP orbit does not close at the stated period: "
            f"{position_error:.6g} km and {velocity_error:.6g} m/s. "
            "Correct its orbit rather than repeatedly resetting a drifting trajectory.")
    one_period = _reference(state, period, cfg)
    def at(t):
        phase = float(t) % period
        return np.asarray(one_period(phase), dtype=float)
    metadata = dict(period_tu=period, period_days=physical_days,
                    position_closure_km=position_error,
                    velocity_closure_m_s=velocity_error,
                    periodicity_verified_in="Earth-Moon CR3BP",
                    resonance="nominal 9:2 NRHO; provenance from supplied seed, not independently reclassified",
                    physical_model_limit="idealized CR3BP, not a mission ephemeris")
    return at, metadata
