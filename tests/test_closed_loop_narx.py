"""Executed-burn NARX data invariants and deterministic replay."""
import numpy as np
from halo_tracking.study import load_config
from halo_tracking.closed_loop_narx import generate_closed_loop_dataset


def test_closed_loop_burn_history_and_tracking(tmp_path):
    cfg = load_config()
    cfg["narx"].update(training_arcs=1, validation_arcs=1, test_arcs=1,
                       arc_duration_periods=.25, decisions_per_period=8,
                       history_steps=1, closed_loop_measurement_cadence_s=86400)
    # Force at least one observer trigger without exceeding its cap.
    cfg["narx"]["observer_guidance"].update(
        trigger_position_km=0.0001, trigger_velocity_m_s=0.0001)
    summary = generate_closed_loop_dataset(cfg, tmp_path)
    assert set(summary) == {"training", "validation", "test"}
    for split in summary:
        with np.load(tmp_path / (split + ".npz")) as data:
            assert data["gateway_tracking_position_error_km"].shape == (1, 3)
            assert data["observer_navigation_normalized"].shape == (1, 3, 6)
            assert data["lso_visible"].shape[-1] == len(cfg["sensors"])
            assert data["observer_history_applied_delta_v_rotating_m_s"].shape == (1, 3, 1, 3)
            assert not data["history_valid"][0]
            assert data["history_valid"][1]
            np.testing.assert_allclose(
                data["observer_history_applied_delta_v_rotating_m_s"][0, 1, 0],
                data["observer_applied_delta_v_rotating_m_s"][0, 0])
            assert np.isfinite(data["gateway_tracking_position_error_km"]).all()


def test_closed_loop_is_reproducible(tmp_path):
    cfg = load_config()
    cfg["narx"].update(training_arcs=1, validation_arcs=1, test_arcs=1,
                       arc_duration_periods=.25, decisions_per_period=8,
                       history_steps=1, closed_loop_measurement_cadence_s=86400)
    generate_closed_loop_dataset(cfg, tmp_path / "first")
    generate_closed_loop_dataset(cfg, tmp_path / "second")
    with np.load(tmp_path / "first" / "training.npz") as a, np.load(tmp_path / "second" / "training.npz") as b:
        for key in ("gateway_navigation_normalized", "observer_applied_delta_v_rotating_m_s",
                    "gateway_tracking_position_error_km"):
            np.testing.assert_equal(a[key], b[key])


def test_mc_initial_conditions_match_covariance_and_reference(tmp_path):
    cfg = load_config()
    cfg["narx"].update(training_arcs=3, validation_arcs=1, test_arcs=1,
                       arc_duration_periods=.25, decisions_per_period=8,
                       history_steps=1, closed_loop_measurement_cadence_s=86400)
    from halo_tracking.orbit_reference import correct_halo_reference
    from halo_tracking.estimation import initial_covariance
    corrected = correct_halo_reference(cfg)
    expected = np.r_[corrected["orbit"]["state"], cfg["gateway"]["state"]]
    generate_closed_loop_dataset(cfg, tmp_path)
    with np.load(tmp_path / "training.npz") as data:
        p = data["initial_covariance_normalized"]
        np.testing.assert_allclose(p[:6, :6], initial_covariance(cfg))
        np.testing.assert_allclose(p[:6, 6:], 0)
        physical = data["initial_physical_offset_normalized"]
        navigation_error = data["initial_navigation_error_normalized"]
        np.testing.assert_allclose(data["initial_truth_normalized"], expected + physical)
        np.testing.assert_allclose(data["initial_navigation_normalized"],
                                   expected + physical + navigation_error)
        np.testing.assert_allclose(data["observer_truth_normalized_evaluation_only"][:, 0],
                                   data["initial_truth_normalized"][:, :6])
        assert np.any(np.abs(physical) > 0)
        assert np.any(np.abs(navigation_error) > 0)
        assert not np.array_equal(physical, navigation_error)
