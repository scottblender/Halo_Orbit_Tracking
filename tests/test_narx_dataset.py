"""Minimal phase-aware dataset invariants, including isolated train/test arcs."""
import json
import numpy as np
import pytest
from halo_tracking.study import load_config
from halo_tracking.narx_dataset import generate_phase_aware_dataset, _optimal_phase


def test_phase_aware_split_and_schema(tmp_path):
    cfg = load_config()
    cfg["narx"].update(training_arcs=2, validation_arcs=1, test_arcs=1,
                       arc_duration_periods=0.25, decisions_per_period=8,
                       history_steps=2)
    summary = generate_phase_aware_dataset(cfg, tmp_path)
    assert set(summary) == {"training", "validation", "test"}
    starts = []
    for split, count in (("training", 2), ("validation", 1), ("test", 1)):
        with np.load(tmp_path / (split + ".npz")) as d:
            assert d["observer_history_error_km_m_s"].shape[0] == count
            assert d["gateway_target_delta_v_rotating_m_s"].shape[-1] == 3
            assert np.all(~d["history_valid"][:1])
            assert np.all(d["history_valid"][1:])
            for name in ("observer",):
                valid = d[name + "_maneuver_valid"]
                required = d[name + "_maneuver_required"]
                burn = d[name + "_target_delta_v_rotating_m_s"]
                assert np.all(burn[~required] == 0)
                assert np.all(np.isnan(burn[~valid]))
                assert np.all(
                    d[name + "_weighted_cost_after"][required & valid]
                    < d[name + "_weighted_cost_before"][required & valid])
            assert not d["gateway_maneuver_required"].any()
            assert not d["gateway_maneuver_valid"].any()
            assert np.isnan(d["gateway_target_delta_v_rotating_m_s"]).all()
            starts.extend(d["observer_truth_normalized_evaluation_only"][:, 0])
    assert len(np.unique(starts, axis=0)) == 4
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["settings"]["gateway_guidance"]["position_tolerance_km"] == 0.1
    assert meta["settings"]["gateway_guidance"]["velocity_tolerance_m_s"] == 0.01
    assert meta["settings"]["observer_guidance"]["position_tolerance_km"] == 100
    assert meta["settings"]["observer_guidance"]["trigger_velocity_m_s"] == 0.5


def test_invalid_phase_rejected(tmp_path):
    cfg = load_config()
    cfg["narx"]["candidate_delay_fractions"] = [0.3]
    cfg["narx"].update(training_arcs=1, validation_arcs=1, test_arcs=1,
                       arc_duration_periods=0.125)
    with pytest.raises(ValueError, match="Candidate burn delays"):
        generate_phase_aware_dataset(cfg, tmp_path)
