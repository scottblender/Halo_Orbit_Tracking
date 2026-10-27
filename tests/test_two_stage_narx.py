"""Small deterministic two-stage test, including no-burn and burn decisions."""
import numpy as np
from halo_tracking.study import load_config
from halo_tracking.two_stage_narx import evaluate_two_stage


def test_two_stage_with_independent_episodes(tmp_path):
    cfg = load_config()
    cfg["narx"].update(arc_duration_periods=2, history_steps=2)
    source = tmp_path / "data"
    source.mkdir()
    period = 1.8572
    times = np.linspace(0, 2 * period, 17)
    for split, n in (("training", 8), ("validation", 3), ("test", 3)):
        rng = np.random.default_rng({"training": 10, "validation": 11, "test": 12}[split])
        arrays = dict(decision_times_tu=times, decision_phase_rad=2 * np.pi * (times / period % 1),
                      history_valid=np.arange(17) >= 2,
                      gateway_visible=np.ones((n, 17), dtype=bool),
                      lso_visible=np.ones((n, 17, 3), dtype=bool),
                      position_covariance_trace_normalized=np.full((n, 17, 2), 1e-6))
        for name in ("observer", "gateway"):
            hist = rng.normal(size=(n, 17, 2, 6))
            actions = rng.normal(size=(n, 17, 2, 3))
            trigger = hist[:, :, -1, 0] > 0
            delta = np.where(trigger[..., None], hist[:, :, -1, :3] * .01, 0.)
            delay = np.where(trigger, 0., 0.)
            arrays.update({
                name + "_history_error_km_m_s": hist,
                name + "_history_applied_delta_v_rotating_m_s": actions,
                name + "_maneuver_required": trigger,
                name + "_maneuver_valid": np.ones((n, 17), dtype=bool),
                name + "_target_delta_v_rotating_m_s": delta,
                name + "_target_burn_delay_tu": delay,
            })
        np.savez_compressed(source / (split + ".npz"), **arrays)
    report = evaluate_two_stage(source, tmp_path / "metrics", cfg)
    assert (tmp_path / "metrics" / "metrics.json").exists()
    for name in ("observer", "gateway"):
        assert report[name]["test"]["evaluated_decisions"] == 45
        assert report[name]["test"]["feasible_burn_examples"] > 0
        assert report[name]["test"]["delta_v_rmse_m_s"] >= 0
        assert 0 <= report[name]["test"]["maneuver_f1"] <= 1
