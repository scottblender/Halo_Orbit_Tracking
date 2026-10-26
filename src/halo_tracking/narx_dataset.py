"""Phase-aware, two-spacecraft inverse-guidance datasets for NARX experiments.

All distances are km internally, all physical velocity errors and impulses m/s.
The two control channels are independent: this does NOT assume an observer burn
can correct Gateway's own orbit. Navigation here is a seeded noisy-state proxy,
not an EKF; replace it with the tracking estimator before flight-like studies.
"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import rhs, propagate
from .frames import state_transform
from .identification import _forward
from .orbit_reference import correct_halo_reference


def _reference(state, duration, cfg):
    sol = solve_ivp(
        lambda t, y: rhs(t, y, cfg["mu"]), (0.0, duration), state,
        dense_output=True, method="DOP853", rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol


def _error(t, state, target, cfg):
    delta = state_transform(t, cfg) @ (state - target)
    return np.r_[delta[:3], 1000.0 * delta[3:]]


def _optimal_phase(t, estimate, reference, cfg, settings):
    """Select a scheduled burn by nonlinear terminal performance.

    Every candidate aims at the SAME terminal epoch. Candidate waiting times
    are nonnegative. A 3-component instantaneous velocity change is solved
    using the STM, then checked by actual nonlinear propagation. Invalid
    candidates are masked, never clipped to the impulse cap.
    """
    period = settings["observer_period_tu"]
    horizon = settings["guidance_horizon_fraction"] * period
    end = t + horizon
    target = reference(end)
    pos_limit = settings["position_tolerance_km"]
    vel_limit = settings["velocity_tolerance_m_s"]
    weight = np.array([1.0 / pos_limit] * 3 + [1.0 / vel_limit] * 3)
    delays = np.asarray(settings["candidate_delay_fractions"], dtype=float) * period
    if np.any(delays < 0) or np.any(delays >= horizon):
        raise ValueError("Candidate burn delays must lie in [0, guidance horizon)")
    if len(delays) == 0:
        raise ValueError("At least one candidate phase is required")
    free, _ = _forward(estimate, t, horizon, cfg, False)
    free_error = _error(end, free, target, cfg)
    free_cost = float(np.linalg.norm(weight * free_error))
    deviation = _error(t, estimate, reference(t), cfg)
    required = bool(
        np.linalg.norm(deviation[:3]) >= settings["trigger_position_km"]
        or np.linalg.norm(deviation[3:]) >= settings["trigger_velocity_m_s"]
    )
    result = dict(required=required, valid=True, delay_tu=0.0,
                  burn=np.zeros(3), before=free_cost, after=free_cost,
                  improvement=0.0)
    if not required:
        return result
    result["valid"] = False
    result["burn"][:] = np.nan
    result["delay_tu"] = np.nan
    result["after"] = np.nan
    result["improvement"] = np.nan
    chosen = None
    velocity_factor = cfg["time_unit_s"] / (1000.0 * cfg["length_unit_km"])
    for delay in delays:
        at_burn = t + float(delay)
        prior, _ = _forward(estimate, t, float(delay), cfg, False) if delay else (estimate, None)
        endpoint, stm = _forward(prior, at_burn, end - at_burn, cfg, True)
        residual = _error(end, endpoint, target, cfg)
        sensitivity = state_transform(end, cfg) @ (stm[:, 3:] * velocity_factor)
        sensitivity[3:] *= 1000.0
        A = weight[:, None] * sensitivity
        b = weight * residual
        reg = settings["regularization"] / settings["maximum_impulse_m_s"]
        burn = np.linalg.lstsq(
            np.vstack((A, reg * np.eye(3))),
            np.r_[-b, np.zeros(3)], rcond=None)[0]
        if not np.all(np.isfinite(burn)) or np.linalg.norm(burn) > settings["maximum_impulse_m_s"]:
            continue
        controlled = prior.copy()
        controlled[3:] += velocity_factor * burn
        final, _ = _forward(controlled, at_burn, end - at_burn, cfg, False)
        after = float(np.linalg.norm(weight * _error(end, final, target, cfg)))
        if not np.isfinite(after) or after >= free_cost:
            continue
        # Equal terminal horizon; use Δv only as a small tie-break penalty.
        score = after + settings["effort_penalty"] * np.linalg.norm(burn)
        if chosen is None or score < chosen[0]:
            chosen = (score, float(delay), burn, after)
    if chosen is not None:
        _, delay, burn, after = chosen
        result.update(valid=True, delay_tu=delay, burn=burn, after=after,
                      improvement=free_cost - after)
    return result


def generate_phase_aware_dataset(cfg, directory):
    """Export independent training/validation/test arcs and phase-selected Δv.

    Labels exist for BOTH the halo observer and Gateway. Nominal Gateway here
    is its specified initial-state CR3BP propagation, not a verified periodic
    orbit. Lagged features are navigation estimates, not hidden truth.
    """
    cfg = deepcopy(cfg)
    s = cfg["narx"]
    for key in ("position_tolerance_km", "velocity_tolerance_m_s",
                "trigger_position_km", "trigger_velocity_m_s",
                "maximum_impulse_m_s", "guidance_horizon_fraction",
                "measurement_position_sigma_km", "measurement_velocity_sigma_m_s"):
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not (s["trigger_position_km"] < s["position_tolerance_km"]
            and s["trigger_velocity_m_s"] < s["velocity_tolerance_m_s"]):
        raise ValueError("Triggers must precede the position/velocity limits")
    if s["history_steps"] < 1 or s["decisions_per_period"] < 2:
        raise ValueError("history_steps and decisions_per_period are invalid")
    delays = np.asarray(s["candidate_delay_fractions"], dtype=float)\n    if delays.ndim != 1 or not len(delays) or not np.all(np.isfinite(delays)) or np.any(delays < 0) or np.any(delays >= s["guidance_horizon_fraction"]):\n        raise ValueError("Candidate burn delays must lie in [0, guidance horizon)")\n    nominal_cfg = correct_halo_reference(cfg)
    period = float(nominal_cfg["orbit"]["period_tu"])
    s["observer_period_tu"] = period
    duration = float(s["arc_duration_periods"]) * period
    if duration <= 0:
        raise ValueError("arc_duration_periods must be positive")
    times = np.linspace(0, duration, round(s["arc_duration_periods"] * s["decisions_per_period"]) + 1)
    ext = duration + s["guidance_horizon_fraction"] * period
    nominal_states = dict(
        observer=np.asarray(nominal_cfg["orbit"]["state"], dtype=float),
        gateway=np.asarray(cfg["gateway"]["state"], dtype=float))
    references = {name: _reference(state, ext, cfg) for name, state in nominal_states.items()}
    converter = np.r_[np.full(3, 1.0 / cfg["length_unit_km"]),
                      np.full(3, cfg["time_unit_s"] / (1000.0 * cfg["length_unit_km"]))]
    noise = np.r_[np.full(3, s["measurement_position_sigma_km"]),
                  np.full(3, s["measurement_velocity_sigma_m_s"])]
    spread = np.r_[np.full(3, s["initial_position_spread_km"]),
                   np.full(3, s["initial_velocity_spread_m_s"])]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(s["seed"]).spawn(3)
    summary = {}
    for split, seed in zip(("training", "validation", "test"), seeds):
        n = int(s[split + "_arcs"])
        if n < 1:
            raise ValueError("Each split must contain at least one arc")
        rng = np.random.default_rng(seed)
        records = []
        for _ in range(n):
            record = {}
            for name, reference in references.items():
                initial = nominal_states[name] + rng.normal(size=6) * spread * converter
                truth = propagate(initial, times, cfg["mu"])
                # Explicitly a noisy navigation proxy, not a sequential EKF.
                navigation = truth + rng.normal(size=truth.shape) * noise * converter
                relative = np.asarray([_error(t, x, reference(t), cfg)
                                       for t, x in zip(times, navigation)])
                labels = [_optimal_phase(t, x, reference, cfg, s)
                          for t, x in zip(times, navigation)]
                prefix = name + "_"
                record[prefix + "truth_normalized_evaluation_only"] = truth
                record[prefix + "navigation_normalized"] = navigation
                record[prefix + "navigation_error_km_m_s"] = relative
                record[prefix + "reference_normalized"] = np.stack([reference(t) for t in times])
                record[prefix + "target_delta_v_rotating_m_s"] = np.stack([q["burn"] for q in labels])
                record[prefix + "target_burn_delay_tu"] = np.array([q["delay_tu"] for q in labels])
                record[prefix + "target_burn_phase_rad"] = np.array([
                    (2.0 * np.pi * ((t + q["delay_tu"]) / period % 1.0)
                     if q["valid"] else np.nan) for t, q in zip(times, labels)])
                for key in ("required", "valid", "before", "after", "improvement"):
                    record[prefix + ("maneuver_" if key in ("required", "valid") else "weighted_cost_") + key] = np.asarray([q[key] for q in labels])
                # Left-pad historical navigation with NaN; mask warm-up points.
                lag = s["history_steps"]
                history = np.full((len(times), lag, 6), np.nan)
                for i in range(lag - 1, len(times)):
                    history[i] = relative[i - lag + 1:i + 1]
                record[prefix + "history_error_km_m_s"] = history
            records.append(record)
        arrays = {key: np.stack([record[key] for record in records])
                  for key in records[0]}
        arrays["decision_times_tu"] = times
        arrays["decision_phase_rad"] = 2 * np.pi * (times / period % 1.0)
        arrays["history_valid"] = np.arange(len(times)) >= s["history_steps"] - 1
        np.savez_compressed(directory / (split + ".npz"), **arrays)
        summary[split] = dict(arcs=n, decisions=n * len(times),
                              observer_valid=int(arrays["observer_maneuver_valid"].sum()),
                              gateway_valid=int(arrays["gateway_maneuver_valid"].sum()))
    metadata = dict(purpose="Phase-selected inverse guidance labels for two independent controlled spacecraft",
                    units="position km; physical velocity and delta-v m/s; phase radians; normalized states CR3BP",
                    settings=s, summary=summary,
                    caveats=["Burn labels are not applied to truth arcs; not closed-loop NARX training.",
                             "Past applied controls are unavailable; history contains navigation lag only.",
                             "Navigation uses a seeded noisy-state proxy, not the operational EKF.",
                             "Gateway reference is an initial-state CR3BP trajectory, not a corrected periodic orbit.",
                             "Valid labels improve weighted terminal error; strict 100 m / velocity-limit terminal satisfaction is not guaranteed.",
                             "No train/validation/test trajectory mixing; fit preprocessing on training only."])
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary
