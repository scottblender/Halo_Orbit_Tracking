"""Behavioral regressions for the single-burn ensemble aggregation."""
import json

import numpy as np
import pytest

from scripts.run_single_burn_monte_carlo import TRIALS, aggregate


@pytest.fixture
def trial_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for trial in range(1, TRIALS + 1):
        day = np.array([0.0, 1.0, 1.0, 365.25])
        if trial == 2:
            day[1:3] += 1.9e-6
        values = np.tile([1.0, 0.05], (day.size, 1))
        if trial == 1:
            values[2, 1] = 0.12
        peak_position, peak_velocity = values.max(axis=0).tolist()
        summary = {
            "trial": trial,
            "seed": trial,
            "completed": True,
            "maximum_position_km": peak_position,
            "maximum_velocity_m_s": peak_velocity,
            "all_evaluated_samples_inside": trial != 1,
            "first_sampled_exit_day": 1.0 if trial == 1 else None,
            "annual_delta_v_m_s": float(trial),
            "burns": 1,
            "initial_departure": [0.0] * 6,
        }
        np.savez(tmp_path / f"trial-{trial:02d}.npz", day=day, values=values,
                 summary=json.dumps(summary))
    return tmp_path


def test_aggregate_includes_post_burn_velocity_violations(trial_files):
    aggregate(trial_files)

    summary = json.loads((trial_files / "benchmark-results/summary.json").read_text())
    assert summary["successful"] == TRIALS - 1
    assert summary["first_mean_exit_day"] is None
    assert summary["first_exits"] == {"1": 1.0}
    assert summary["peaks_of_mean"] == pytest.approx([1.0, (0.12 + 0.05 * (TRIALS - 1)) / TRIALS])
    assert summary["mean_delta_v_m_s"] == pytest.approx((TRIALS + 1) / 2)


@pytest.mark.parametrize(("epochs", "message"), [
    ([0.0, 1.001, 1.001, 365.25], "maximum epoch shift"),
    ([0.0, 1.0, 1.000001, 365.25], "sample layout differs"),
    ([0.0, 1.0, 0.9, 365.25], "epochs must be nondecreasing"),
    ([0.0, 1.0, 1.0, 360.0], "invalid propagation epochs"),
])
def test_aggregate_rejects_mismatched_epoch_grids(trial_files, epochs, message):
    path = trial_files / "trial-02.npz"
    with np.load(path) as archive:
        record = dict(archive)
    record["day"] = np.array(epochs)
    np.savez(path, **record)

    with pytest.raises(ValueError, match=message):
        aggregate(trial_files)
