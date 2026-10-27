"""Periodic Gateway, common-noise, and closed-loop NARX rollout invariants."""
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

from halo_tracking.study import load_config
from halo_tracking.gateway_reference_periodic import gateway_periodic_reference
from halo_tracking.year_study import (
    _observer_reference, _initial_conditions, simulate_one_arc,
    run_paired_study,
)
from halo_tracking.two_stage_narx import predict_observer_policy


def _never_burn(history_steps=1):
    X = np.zeros((3, 9 * history_steps + 5))
    return dict(classifier=DummyClassifier(strategy="constant", constant=0).fit(X, np.zeros(3)),
                regressors=[DummyRegressor().fit(X, np.zeros(3)) for _ in range(3)],
                timing=DummyClassifier(strategy="constant", constant=0).fit(X, np.zeros(3)),
                history_steps=history_steps, feature_size=X.shape[1],
                candidate_delay_fractions=[0., .0625, .125, .1875],
                fitted_training_decisions=3, fitted_feasible_burns=1,
                training_only=True)


def test_gateway_seed_closes_and_periodic_phase_wraps():
    cfg = load_config()
    ref, diagnostic = gateway_periodic_reference(cfg)
    assert diagnostic["position_closure_km"] < .001
    assert diagnostic["velocity_closure_m_s"] < .001
    np.testing.assert_allclose(ref(0), ref(cfg["gateway"]["period_tu"]),
                               rtol=0, atol=1e-12)
    cfg["gateway"]["period_tu"] *= 1.01
    import pytest
    with pytest.raises(ValueError, match="does not close"):
        gateway_periodic_reference(cfg)


def test_no_burn_predictor():
    policy = _never_burn()
    decision = predict_observer_policy(policy, np.zeros(14), 1.85, 10.)
    assert not decision["required"]
    np.testing.assert_array_equal(decision["burn"], 0.)


def test_paired_rollouts_use_same_perturbations_and_never_burn_gateway(tmp_path):
    cfg = load_config()
    cfg["narx"]["history_steps"] = 1
    model = _never_burn()
    # Very short deterministic integration exercises all three controllers
    # without requiring a 365-day run inside routine CI.
    summary = run_paired_study(cfg, tmp_path, model, days=.12, arcs=1,
                               measurement_cadence_s=3600., decisions_per_period=8)
    assert summary["paired_noise"]
    assert set(summary["aggregate"]) == {"uncontrolled", "stm", "narx"}
    for name in summary["aggregate"]:
        with np.load(tmp_path / f"{name}_arc_000.npz") as d:
            assert d["decision_times_tu"].shape[0] == 3
            assert np.isfinite(d["gateway_tracking_position_error_km"]).all()
            assert d["applied_delta_v_rotating_m_s"].shape == (3, 3)
    with np.load(tmp_path / "uncontrolled_arc_000.npz") as d:
        np.testing.assert_array_equal(d["applied_delta_v_rotating_m_s"], 0.)
    with np.load(tmp_path / "narx_arc_000.npz") as d:
        np.testing.assert_array_equal(d["applied_delta_v_rotating_m_s"], 0.)
