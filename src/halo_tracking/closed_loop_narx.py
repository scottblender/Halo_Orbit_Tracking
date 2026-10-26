"""Closed-loop dual-spacecraft guidance episodes with angle-only EKF tracking.

Each decision uses estimated states, not truth. Scheduled burns modify truth
and the estimator prediction, and their executed values are exported as past
control inputs for autoregressive nonlinear regression.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .dynamics import propagate_step
from .estimation import initial_covariance
from .frames import state_transform
from .gateway_comparison import (
    gateway_visible, spacecraft_jacobians, spacecraft_measurement, update,
)
from .identification import _impulse_label
from .measurements import ARCSEC, measurement_jacobian, predict, wrap
from .narx_dataset import _reference, _error, _optimal_phase
from .network import sensor_configs
from .orbit_reference import correct_halo_reference
from .surface_sensor import elevation
from .visibility import optical_visible
from .surface_sensor import site_rotating


def _sample_observations(t, observer, gateway, cfg, rng, sites):
    """Sample accepted LSO and observer-to-Gateway angle pairs."""
    lso = np.full((len(sites), 2), np.nan)
    visible = np.zeros(len(sites), dtype=bool)
    sigma_l = cfg["measurement"]["angle_sigma_arcsec"] * ARCSEC
    for j, local in enumerate(sites):
        station = np.array([1 - cfg["mu"], 0., 0.]) + site_rotating(local)
        accepted = (elevation(observer, local) >=
                    np.deg2rad(local["site"]["minimum_elevation_deg"])
                    and optical_visible(station, observer[:3], t, local, link="lso"))
        if accepted:
            lso[j] = predict(t, observer, local) + rng.normal(0, sigma_l, 2)
            lso[j, 0] = wrap(lso[j, 0])
            visible[j] = True
    camera = np.full(2, np.nan)
    target_visible = gateway_visible(observer, gateway, cfg, t)
    if target_visible:
        sigma_g = cfg["comparison"]["gateway_camera_sigma_arcsec"] * ARCSEC
        camera = spacecraft_measurement(t, observer, gateway, cfg) + rng.normal(0, sigma_g, 2)
        camera[0] = wrap(camera[0])
    return lso, visible, camera, bool(target_visible)


def _ekf_observations(t, x, P, lso, visible, camera, target_visible, cfg, sites):
    Rl = np.eye(2) * (cfg["measurement"]["angle_sigma_arcsec"] * ARCSEC)**2
    Rg = np.eye(2) * (cfg["comparison"]["gateway_camera_sigma_arcsec"] * ARCSEC)**2
    for j, local in enumerate(sites):
        if visible[j]:
            H = np.c_[measurement_jacobian(t, x[:6], local, False), np.zeros((2, 6))]
            innovation = lso[j] - predict(t, x[:6], local, False)
            x, P = update(x, P, H, Rl, innovation)
    if target_visible:
        Ho, Hg = spacecraft_jacobians(t, x[:6], x[6:], cfg)
        innovation = camera - spacecraft_measurement(t, x[:6], x[6:], cfg)
        x, P = update(x, P, np.c_[Ho, Hg], Rg, innovation, consider_observer=True)
    return x, P


def generate_closed_loop_dataset(cfg, directory):
    """Generate split-isolated applied-burn trajectories, navigation, and labels.

    Decisions are on a fixed grid. An accepted candidate can be scheduled
    between grid points; the next guidance decision is suppressed until it
    has executed. The same known burn is applied to truth and estimator.
    """
    cfg = deepcopy(cfg)
    s = cfg["narx"]
    Tcfg = correct_halo_reference(cfg)
    T = float(Tcfg["orbit"]["period_tu"])
    settings = {**s, "observer_period_tu": T}
    decisions = int(round(s["arc_duration_periods"] * s["decisions_per_period"]))
    if decisions < 2:
        raise ValueError("At least two decision intervals are required")
    horizon = s["arc_duration_periods"] * T
    grid = np.linspace(0, horizon, decisions + 1)
    sensor_step = s.get("closed_loop_measurement_cadence_s", 21600) / cfg["time_unit_s"]
    if sensor_step <= 0:
        raise ValueError("Measurement cadence must be positive")
    sensor_times = np.unique(np.r_[np.arange(0, horizon, sensor_step), grid])
    initial_ref = {
        "observer": np.asarray(Tcfg["orbit"]["state"], dtype=float),
        "gateway": np.asarray(cfg["gateway"]["state"], dtype=float),
    }
    reference = {name: _reference(state, horizon + T, cfg)
                 for name, state in initial_ref.items()}
    scales = np.r_[np.full(3, 1 / cfg["length_unit_km"]),
                   np.full(3, cfg["time_unit_s"] / (1000 * cfg["length_unit_km"]))]
    spread = scales * np.r_[np.full(3, s["initial_position_spread_km"]),
                            np.full(3, s["initial_velocity_spread_m_s"])]
    sites = sensor_configs(cfg)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(s["seed"]).spawn(3)
    summary = {}
    for split, stream in zip(("training", "validation", "test"), streams):
        n = int(s[split + "_arcs"])
        if n < 1:
            raise ValueError("Each split requires one or more arcs")
        rng = np.random.default_rng(stream)
        episodes = []
        for arc in range(n):
            truth = np.r_[initial_ref["observer"] + rng.normal(size=6) * spread,
                          initial_ref["gateway"] + rng.normal(size=6) * spread]
            prior_o = initial_covariance(cfg)
            gateway_cfg = deepcopy(cfg)
            gateway_cfg["estimation"]["initial_position_sigma_km"] = cfg["comparison"]["gateway_position_sigma_km"]
            gateway_cfg["estimation"]["initial_velocity_sigma_km_s"] = cfg["comparison"]["gateway_velocity_sigma_km_s"]
            prior_g = initial_covariance(gateway_cfg)
            x = truth + np.r_[rng.multivariate_normal(np.zeros(6), prior_o),
                               rng.multivariate_normal(np.zeros(6), prior_g)]
            P = np.zeros((12, 12))
            P[:6, :6], P[6:, 6:] = prior_o, prior_g
            scheduled = {"observer": None, "gateway": None}
            executed = {"observer": np.zeros(3), "gateway": np.zeros(3)}
            results = {name: {key: [] for key in (
                "truth_normalized_evaluation_only", "navigation_normalized",
                "navigation_error_km_m_s", "reference_normalized",
                "target_delta_v_rotating_m_s", "target_burn_delay_tu",
                "target_burn_phase_rad", "maneuver_required", "maneuver_valid",
                "applied_delta_v_rotating_m_s", "tracking_position_error_km",
                "tracking_velocity_error_m_s", "scheduled_pending")}
                for name in initial_ref}
            observed = []
            lso_visible = []
            gateway_visible_history = []
            previous = 0.
            all_times = np.unique(np.r_[sensor_times, grid])
            index = 0
            def advance(t):
                nonlocal truth, x, P, previous
                if t <= previous:
                    return
                yo, Fo, Qo = propagate_step(x[:6], previous, t, cfg["mu"], cfg["estimation"]["acceleration_psd_normalized"])
                yg, Fg, Qg = propagate_step(x[6:], previous, t, cfg["mu"], 0.)
                xt = np.r_[yo, yg]
                F = np.zeros((12, 12))
                F[:6, :6], F[6:, 6:] = Fo, Fg
                Q = np.zeros((12, 12))
                Q[:6, :6], Q[6:, 6:] = Qo, Qg
                P = F @ P @ F.T + Q
                truth = np.r_[propagate_step(truth[:6], previous, t, cfg["mu"])[0],
                              propagate_step(truth[6:], previous, t, cfg["mu"])[0]]
                x = xt
                previous = t
            def execute(name):
                pending = scheduled[name]
                if pending is None:
                    return
                burn = pending[1]
                sl = slice(0, 6) if name == "observer" else slice(6, 12)
                truth[sl][3:] += burn * scales[3:]
                x[sl][3:] += burn * scales[3:]
                executed[name] += burn
                scheduled[name] = None

            while index < len(all_times):
                t = float(all_times[index])
                while True:
                    due = [(name, task) for name, task in scheduled.items()
                           if task is not None and task[0] <= t + 1e-12]
                    if not due:
                        break
                    name, task = min(due, key=lambda item: item[1][0])
                    advance(task[0])
                    execute(name)
                advance(t)
                lso, vis, camera, gv = _sample_observations(t, truth[:6], truth[6:], cfg, rng, sites)
                x, P = _ekf_observations(t, x, P, lso, vis, camera, gv, cfg, sites)
                if np.any(np.isclose(grid, t, atol=1e-12, rtol=0)):
                    observed.append(np.array([np.trace(P[:3, :3]), np.trace(P[6:9, 6:9])]))
                    lso_visible.append(vis.copy())
                    gateway_visible_history.append(gv)
                    for name, sl in (("observer", slice(0, 6)), ("gateway", slice(6, 12))):
                        physical = _error(t, x[sl], reference[name](t), cfg)
                        tracking = _error(t, x[sl], truth[sl], cfg)
                        policy = {**settings, **s[name + "_guidance"]}
                        if scheduled[name] is None:
                            label = _optimal_phase(t, x[sl], reference[name], cfg, policy)
                        else:
                            label = dict(required=False, valid=False, burn=np.full(3, np.nan), delay_tu=np.nan)
                        for key, value in (
                            ("truth_normalized_evaluation_only", truth[sl].copy()),
                            ("navigation_normalized", x[sl].copy()),
                            ("navigation_error_km_m_s", physical),
                            ("reference_normalized", reference[name](t)),
                            ("tracking_position_error_km", np.linalg.norm(tracking[:3])),
                            ("tracking_velocity_error_m_s", np.linalg.norm(tracking[3:])),
                            ("target_delta_v_rotating_m_s", label["burn"].copy()),
                            ("target_burn_delay_tu", label["delay_tu"]),
                            ("target_burn_phase_rad", (2 * np.pi * ((t + label["delay_tu"]) / T % 1)) if label["valid"] else np.nan),
                            ("maneuver_required", label["required"]),
                            ("maneuver_valid", label["valid"]),
                            ("applied_delta_v_rotating_m_s", executed[name].copy()),
                            ("scheduled_pending", scheduled[name] is not None),
                        ):
                            results[name][key].append(value)
                        if label["required"] and label["valid"]:
                            scheduled[name] = (t + label["delay_tu"], label["burn"].copy())
                            if label["delay_tu"] == 0:
                                execute(name)
                    # Record instantaneous commands at the current epoch, too.
                    for name in initial_ref:
                        results[name]["applied_delta_v_rotating_m_s"][-1] = executed[name].copy()
                    executed = {"observer": np.zeros(3), "gateway": np.zeros(3)}
                index += 1
            episode = {}
            for name in initial_ref:
                for key, values in results[name].items():
                    episode[name + "_" + key] = np.asarray(values)
                lag = int(s["history_steps"])
                deviations = episode[name + "_navigation_error_km_m_s"]
                actions = episode[name + "_applied_delta_v_rotating_m_s"]
                nav_lag = np.full((len(grid), lag, 6), np.nan)
                action_lag = np.full((len(grid), lag, 3), np.nan)
                for k in range(lag, len(grid)):
                    nav_lag[k] = deviations[k - lag + 1:k + 1]
                    # The current action is a target, never an input feature.
                    action_lag[k] = actions[k - lag:k]
                episode[name + "_history_error_km_m_s"] = nav_lag
                episode[name + "_history_applied_delta_v_rotating_m_s"] = action_lag
            episode["lso_visible"] = np.asarray(lso_visible)
            episode["gateway_visible"] = np.asarray(gateway_visible_history)
            episode["position_covariance_trace_normalized"] = np.asarray(observed)
            episodes.append(episode)
        arrays = {key: np.stack([e[key] for e in episodes]) for key in episodes[0]}
        arrays["decision_times_tu"] = grid
        arrays["decision_phase_rad"] = 2 * np.pi * (grid / T % 1)
        arrays["history_valid"] = np.arange(len(grid)) >= int(s["history_steps"])
        np.savez_compressed(out / (split + ".npz"), **arrays)
        summary[split] = {"arcs": n, "decision_epochs_per_arc": len(grid),
                          "gateway_position_within_100m_fraction": float(np.mean(arrays["gateway_tracking_position_error_km"] <= .1)),
                          "gateway_velocity_within_limit_fraction": float(np.mean(arrays["gateway_tracking_velocity_error_m_s"] <= s["gateway_guidance"]["velocity_tolerance_m_s"]))}
    (out / "metadata.json").write_text(json.dumps(dict(
        purpose="Executed-control trajectories for phase-aware NARX with two-body angle-only EKF",
        assumptions="Scheduled guidance candidates are applied to both truth and estimator; no execution error model.",
        interpretation="Gateway tracking metrics compare EKF estimate against simulated Gateway truth; guidance uses reference departure.",
        configuration=s, summary=summary), indent=2) + "\n")
    return summary
