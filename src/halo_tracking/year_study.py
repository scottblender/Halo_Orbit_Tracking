"""Paired uncontrolled / STM / learned-NARX year-long observer station-keeping.

All three policies receive identical initial physical/navigation perturbations
and epoch-indexed independent standard-normal optical errors. Optical
acceptance depends on each policy's real geometry, but a common source draw
is used at every station and epoch. Gateway follows a closure-verified 9:2
periodic CR3BP reference with no maneuvers.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from .closed_loop_narx import _ekf_observations, _sample_observations
from .dynamics import propagate_step
from .estimation import initial_covariance
from .gateway_reference_periodic import gateway_periodic_reference
from .narx_dataset import _error, _optimal_phase, _reference
from .network import sensor_configs
from .orbit_reference import correct_halo_reference
from .two_stage_narx import predict_observer_policy


def _observer_reference(cfg):
    corrected = correct_halo_reference(cfg)
    period = float(corrected["orbit"]["period_tu"])
    one_cycle = _reference(np.asarray(corrected["orbit"]["state"], float), period, cfg)
    def at(t):
        return np.asarray(one_cycle(float(t) % period), dtype=float)
    return at, period, corrected


def _initial_conditions(cfg, observer_initial, gateway_initial, seed, n_observations, n_sites):
    """Generate an arc-independent, policy-independent source of randomness."""
    rng = np.random.default_rng(seed)
    prior_o = initial_covariance(cfg)
    gc = deepcopy(cfg)
    gc["estimation"]["initial_position_sigma_km"] = cfg["comparison"]["gateway_position_sigma_km"]
    gc["estimation"]["initial_velocity_sigma_km_s"] = cfg["comparison"]["gateway_velocity_sigma_km_s"]
    prior_g = initial_covariance(gc)
    P0 = np.zeros((12, 12))
    P0[:6, :6] = prior_o
    P0[6:, 6:] = prior_g
    physical = rng.multivariate_normal(np.zeros(6), prior_o)
    estimate_noise = rng.multivariate_normal(np.zeros(12), P0)
    truth = np.r_[observer_initial + physical, gateway_initial]
    estimate = truth + estimate_noise
    optical_noise = rng.standard_normal((n_observations, n_sites, 2))
    camera_noise = rng.standard_normal((n_observations, 2))
    return truth, estimate, P0, optical_noise, camera_noise


def simulate_one_arc(cfg, controller, observer_reference, observer_period,
                     gateway_reference, decision_times, measurement_times,
                     initial, model=None, max_departure_km=20000.):
    """Roll out a policy using actual post-burn LSO/Gateway EKF measurements.

    Returns full time-aligned arrays. Missing observations after a numerical
    failure or large departure remain NaN and are counted as NOT successful.
    """
    if controller not in ("uncontrolled", "stm", "narx"):
        raise ValueError("Unknown controller")
    if controller == "narx" and model is None:
        raise ValueError("NARX rollout requires a fitted training-only model")
    if max_departure_km <= 0:
        raise ValueError("max_departure_km must be positive")
    times = np.asarray(decision_times, dtype=float)
    sensors = np.asarray(measurement_times, dtype=float)
    truth, x, P, optical_noise, camera_noise = initial
    truth, x, P = truth.copy(), x.copy(), P.copy()
    n = len(times)
    sites = sensor_configs(cfg)
    s = cfg["narx"]
    settings = {**s, **s["observer_guidance"], "observer_period_tu": observer_period}
    velocity_scale = cfg["time_unit_s"] / (1000 * cfg["length_unit_km"])
    L = int(s["history_steps"])
    records = {
        "observer_truth_position_departure_km": np.full(n, np.nan),
        "observer_truth_velocity_departure_m_s": np.full(n, np.nan),
        "observer_navigation_position_error_km": np.full(n, np.nan),
        "gateway_tracking_position_error_km": np.full(n, np.nan),
        "gateway_tracking_velocity_error_m_s": np.full(n, np.nan),
        "observer_covariance_trace_normalized": np.full(n, np.nan),
        "gateway_covariance_trace_normalized": np.full(n, np.nan),
        "lso_visible_count": np.full(n, np.nan),
        "gateway_visible": np.full(n, np.nan),
        "maneuver_requested": np.zeros(n, dtype=bool),
        "maneuver_rejected": np.zeros(n, dtype=bool),
        "predicted_delta_v_rotating_m_s": np.full((n, 3), np.nan),
        "predicted_delay_tu": np.full(n, np.nan),
        "predicted_valid": np.zeros(n, dtype=bool),
        "model_invoked": np.zeros(n, dtype=bool),
        "applied_delta_v_rotating_m_s": np.zeros((n, 3)),
    }
    nav_history = []
    applied = records["applied_delta_v_rotating_m_s"]
    previous = 0.
    pending = None
    executed = np.zeros(3)
    halted = None
    k = 0
    # The same pre-sampled noise arrays are indexed by epoch, not by accepted
    # observations; divergent policies cannot consume the RNG differently.
    def advance(t):
        nonlocal truth, x, P, previous
        if t <= previous:
            return
        xo, Fo, Qo = propagate_step(x[:6], previous, t, cfg["mu"],
                                    cfg["estimation"]["acceleration_psd_normalized"])
        xg, Fg, Qg = propagate_step(x[6:], previous, t, cfg["mu"], 0.)
        F = np.zeros((12, 12))
        F[:6, :6], F[6:, 6:] = Fo, Fg
        Q = np.zeros((12, 12))
        Q[:6, :6], Q[6:, 6:] = Qo, Qg
        P = F @ P @ F.T + Q
        observer_truth = propagate_step(truth[:6], previous, t, cfg["mu"])[0]
        truth = np.r_[observer_truth, gateway_reference(t)]
        x = np.r_[xo, xg]
        previous = t

    def execute():
        nonlocal pending, executed
        if pending is not None:
            burn = pending[1]
            truth[3:6] += velocity_scale * burn
            x[3:6] += velocity_scale * burn
            executed += burn
            pending = None

    for i, t in enumerate(sensors):
        t = float(t)
        try:
            if pending is not None and pending[0] <= t + 1e-12:
                advance(float(pending[0]))
                execute()
            advance(t)
            obs = _sample_observations(
                t, truth[:6], truth[6:], cfg, None, sites,
                paired_noise=(optical_noise[i], camera_noise[i]))
            lso, visible, camera, target_visible = obs
            x, P = _ekf_observations(
                t, x, P, lso, visible, camera, target_visible, cfg, sites)
            if not np.isfinite(x).all() or not np.isfinite(P).all():
                raise FloatingPointError("Nonfinite EKF state or covariance")
            if k >= n or abs(t - times[k]) > 1e-10:
                continue
            physical = _error(t, truth[:6], observer_reference(t), cfg)
            nav = _error(t, x[:6], observer_reference(t), cfg)
            nav_true = _error(t, x[:6], truth[:6], cfg)
            gw_error = _error(t, x[6:], truth[6:], cfg)
            pos = np.linalg.norm(physical[:3])
            records["observer_truth_position_departure_km"][k] = pos
            records["observer_truth_velocity_departure_m_s"][k] = np.linalg.norm(physical[3:])
            records["observer_navigation_position_error_km"][k] = np.linalg.norm(nav_true[:3])
            records["gateway_tracking_position_error_km"][k] = np.linalg.norm(gw_error[:3])
            records["gateway_tracking_velocity_error_m_s"][k] = np.linalg.norm(gw_error[3:])
            records["observer_covariance_trace_normalized"][k] = np.trace(P[:3, :3])
            records["gateway_covariance_trace_normalized"][k] = np.trace(P[6:9, 6:9])
            records["lso_visible_count"][k] = int(visible.sum())
            records["gateway_visible"][k] = float(target_visible)
            applied[k] += executed
            executed[:] = 0.
            nav_history.append(nav)
            if pos > max_departure_km:
                halted = "departure_cutoff"
                k += 1
                break
            if pending is None and controller == "stm":
                command = _optimal_phase(t, x[:6], observer_reference, cfg, settings)
            elif pending is None and controller == "narx" and k >= L:
                feature = np.r_[np.asarray(nav_history[k-L+1:k+1]).reshape(-1),
                                applied[k-L:k].reshape(-1),
                                np.sin(2*np.pi*(t/observer_period % 1)),
                                np.cos(2*np.pi*(t/observer_period % 1)),
                                float(target_visible), float(visible.sum()),
                                np.log10(max(np.trace(P[:3, :3]), 1e-30))]
                records["model_invoked"][k] = True
                command = predict_observer_policy(
                    model, feature, observer_period,
                    settings["maximum_impulse_m_s"])
                records["predicted_valid"][k] = bool(command["valid"])
                if command["required"]:
                    records["predicted_delta_v_rotating_m_s"][k] = command["burn"]
                    records["predicted_delay_tu"][k] = command["delay_tu"]
            else:
                command = dict(required=False, valid=True,
                               burn=np.zeros(3), delay_tu=0.)
            if command["required"]:
                records["maneuver_requested"][k] = True
                if not command["valid"] or t + command["delay_tu"] > times[-1] + 1e-12:
                    records["maneuver_rejected"][k] = True
                else:
                    pending = (t + command["delay_tu"], command["burn"].copy())
                    if command["delay_tu"] == 0:
                        execute()
                        applied[k] += executed
                        executed[:] = 0.
            k += 1
        except (ValueError, OverflowError, FloatingPointError, RuntimeError,
                np.linalg.LinAlgError, ZeroDivisionError) as exc:
            halted = f"{type(exc).__name__}: {exc}"
            break
    records["decision_times_tu"] = times
    records["day"] = times * cfg["time_unit_s"] / 86400.
    records["completed_full_span"] = bool(k == n and halted is None)
    records["halted_reason"] = halted
    return records


def _one_arc_summary(episode, cfg, controller, arc):
    s = cfg["narx"]["observer_guidance"]
    p = episode["observer_truth_position_departure_km"]
    v = episode["observer_truth_velocity_departure_m_s"]
    gp = episode["gateway_tracking_position_error_km"]
    gv = episode["gateway_tracking_velocity_error_m_s"]
    valid = np.isfinite(p) & np.isfinite(v)
    corridor = valid & (p <= s["position_tolerance_km"]) & (
        v <= s["velocity_tolerance_m_s"])
    valid_g = np.isfinite(gp) & np.isfinite(gv)
    within_g = valid_g & (gp <= 0.1) & (gv <= 0.01)
    impulses = episode["applied_delta_v_rotating_m_s"]
    count = np.count_nonzero(np.linalg.norm(impulses, axis=1) > 1e-12)
    def finite_max(x):
        return float(np.max(x[np.isfinite(x)])) if np.isfinite(x).any() else None
    def finite_rmse(x):
        return float(np.sqrt(np.mean(x[np.isfinite(x)]**2))) if np.isfinite(x).any() else None
    failed = np.flatnonzero(~corridor)
    return dict(controller=controller, arc=arc,
                completed_full_span=episode["completed_full_span"],
                maintained_corridor_full_year=bool(episode["completed_full_span"] and corridor.all()),
                first_corridor_exit_day=(float(episode["day"][failed[0]]) if len(failed) else None),
                evaluated_decisions=int(valid.sum()), planned_decisions=len(p),
                corridor_fraction=float(corridor.mean()),
                maximum_observer_departure_km=finite_max(p),
                maximum_observer_velocity_departure_m_s=finite_max(v),
                delta_v_total_m_s=float(np.linalg.norm(impulses, axis=1).sum()),
                executed_burn_records=int(count),
                requested_burns=int(episode["maneuver_requested"].sum()),
                rejected_burns=int(episode["maneuver_rejected"].sum()),
                gateway_position_rmse_km=finite_rmse(gp),
                gateway_velocity_rmse_m_s=finite_rmse(gv),
                gateway_tracking_joint_compliance_fraction=float(within_g.mean()),
                gateway_tracking_valid_fraction=float(valid_g.mean()),
                gateway_visibility_fraction=float(np.nansum(episode["gateway_visible"])/len(p)),
                lso_visible_fraction=float(np.nansum(episode["lso_visible_count"] > 0)/len(p)),
                halted_reason=episode["halted_reason"],
                model_calls=int(episode["model_invoked"].sum()),
                invalid_model_commands=int(np.sum(episode["model_invoked"] & episode["maneuver_rejected"])))


def run_paired_study(cfg, output_dir, bundle, days=365, arcs=5, seed=24017,
                     measurement_cadence_s=21600, decisions_per_period=8,
                     max_departure_km=20000.):
    """Run paired policy rollouts and save independently inspectable outputs.

    The same initial states and complete epoch-indexed optical noise arrays
    are used for all three policies within each arc. No test arc is trained on.
    """
    if days <= 0 or arcs < 1 or measurement_cadence_s <= 0 or decisions_per_period < 2:
        raise ValueError("Invalid rollout span, samples, or cadence")
    teacher_decisions = int(cfg["narx"]["decisions_per_period"])
    teacher_cadence = float(cfg["narx"].get("closed_loop_measurement_cadence_s", 21600.))
    if decisions_per_period != teacher_decisions or not np.isclose(measurement_cadence_s, teacher_cadence):
        raise ValueError(
            f"Training/deployment cadence mismatch: teacher {teacher_decisions} decisions/period, "
            f"{teacher_cadence:g} s measurement cadence; deployment "
            f"{decisions_per_period} decisions/period, {measurement_cadence_s:g} s. "
            "Regenerate matching teacher data or match the rollout settings.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observer_at, observer_period, corrected = _observer_reference(cfg)
    gateway_at, gateway_meta = gateway_periodic_reference(cfg)
    duration_tu = days * 86400. / cfg["time_unit_s"]
    decision_count = max(2, int(np.ceil(duration_tu / observer_period * decisions_per_period)))
    grid = np.linspace(0., duration_tu, decision_count + 1)
    measurement_times = np.unique(np.r_[np.arange(
        0., duration_tu, measurement_cadence_s / cfg["time_unit_s"]), grid])
    locations = sensor_configs(cfg)
    results = []
    for arc in range(arcs):
        initial = _initial_conditions(
            cfg, np.asarray(corrected["orbit"]["state"]), gateway_at(0),
            seed + arc, len(measurement_times), len(locations))
        for controller in ("uncontrolled", "stm", "narx"):
            episode = simulate_one_arc(
                cfg, controller, observer_at, observer_period, gateway_at,
                grid, measurement_times, initial, model=bundle,
                max_departure_km=max_departure_km)
            arrays = {k: v for k, v in episode.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(output_dir / f"{controller}_arc_{arc:03d}.npz", **arrays)
            summary = _one_arc_summary(episode, cfg, controller, arc)
            results.append(summary)
            print(f"arc {arc+1}/{arcs} {controller}: "
                  f"completed={summary['completed_full_span']}, "
                  f"corridor={summary['corridor_fraction']:.2%}, "
                  f"delta_v={summary['delta_v_total_m_s']:.3f} m/s",
                  flush=True)
    with (output_dir / "arc_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    aggregate = {}
    for policy in ("uncontrolled", "stm", "narx"):
        selected = [r for r in results if r["controller"] == policy]
        aggregate[policy] = {
            "arcs": len(selected),
            "full_span_completion_rate": float(np.mean([
                r["completed_full_span"] for r in selected])),
            "full_year_corridor_success_rate": float(np.mean([
                r["maintained_corridor_full_year"] for r in selected])),
            "median_total_delta_v_m_s": float(np.median([
                r["delta_v_total_m_s"] for r in selected])),
            "median_gateway_position_rmse_km": float(np.median([
                r["gateway_position_rmse_km"] for r in selected
                if r["gateway_position_rmse_km"] is not None]))
                if any(r["gateway_position_rmse_km"] is not None for r in selected) else None,
        }
    payload = dict(days=days, arcs=arcs, seed=seed, controllers=list(aggregate),
                   paired_noise=True, observer_reference_closure=corrected["orbit"].get("correction"),
                   gateway_reference_validation=gateway_meta,
                   model_training_decisions=bundle["fitted_training_decisions"],
                   model_training_feasible_burns=bundle["fitted_feasible_burns"],
                   measurement_cadence_s=measurement_cadence_s,
                   decisions_per_period=decisions_per_period,
                   failure_rule="All unevaluated epochs count as unsuccessful; excursions beyond the cutoff halt integration",
                   notes=["CR3BP-only prescribed Gateway reference; no Gateway station-keeping",
                          "STM is the teacher policy; NARX predictions are executed without STM fallback",
                          "Initial states and optical noise are paired across controller alternatives",
                          "This is an idealized dynamical and sensor-model experiment"],
                   aggregate=aggregate, arc_metrics=results)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
