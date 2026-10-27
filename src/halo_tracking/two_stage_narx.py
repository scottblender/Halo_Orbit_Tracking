"""Two-stage baseline: maneuver classifier, then delta-v regression and delay class.

All features are available at decision time. Trajectories, not rows, are split.
This is offline imitation of the STM policy, not closed-loop learned control.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def _load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _features(d, name):
    history = d[name + "_history_error_km_m_s"]
    actions = d[name + "_history_applied_delta_v_rotating_m_s"]
    n, steps = history.shape[:2]
    phase = np.broadcast_to(d["decision_phase_rad"][None, :, None], (n, steps, 1))
    visible = d["gateway_visible"][..., None]
    lso_count = d["lso_visible"].sum(axis=-1)[..., None]
    covariance = d["position_covariance_trace_normalized"]
    # Position trace for observer/Gateway respectively, available from EKF.
    covariance = covariance[..., 0 if name == "observer" else 1, None]
    x = np.concatenate((history.reshape(n, steps, -1),
                        actions.reshape(n, steps, -1),
                        np.sin(phase), np.cos(phase),
                        visible.astype(float), lso_count.astype(float),
                        np.log10(np.maximum(covariance, 1e-30))), axis=-1)
    valid = np.broadcast_to(d["history_valid"][None, :], (n, steps))
    return x.reshape(-1, x.shape[-1]), valid.ravel()


def _timing_classes(delays, candidates, period):
    classes = np.full(len(delays), -1, dtype=int)
    for i, frac in enumerate(candidates):
        matches = np.isfinite(delays) & np.isclose(delays, frac * period, atol=1e-8, rtol=1e-7)
        classes[matches] = i
    return classes


def _classifier(x, y, seed):
    if len(np.unique(y)) == 1:
        return DummyClassifier(strategy="constant", constant=int(y[0])).fit(x, y)
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                          min_samples_leaf=5, random_state=seed).fit(x, y)


def evaluate_two_stage(dataset_dir, output_dir, cfg):
    """Fit on training episodes, inspect validation, evaluate test once."""
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    splits = {key: _load(dataset_dir / (key + ".npz"))
              for key in ("training", "validation", "test")}
    output_dir.mkdir(parents=True, exist_ok=True)
    setting = cfg["narx"]
    period = float(splits["training"]["decision_times_tu"][-1]) / setting["arc_duration_periods"]
    report = {}
    for name in ("observer",):
        X, mask, need, valid, burn, timing = {}, {}, {}, {}, {}, {}
        for split, d in splits.items():
            X[split], mask[split] = _features(d, name)
            need[split] = d[name + "_maneuver_required"].ravel().astype(bool)
            valid[split] = d[name + "_maneuver_valid"].ravel().astype(bool)
            burn[split] = d[name + "_target_delta_v_rotating_m_s"].reshape(-1, 3)
            timing[split] = _timing_classes(
                d[name + "_target_burn_delay_tu"].ravel(),
                setting["candidate_delay_fractions"], period)
            mask[split] &= np.isfinite(X[split]).all(axis=1)
        train = mask["training"]
        if not train.any():
            raise ValueError("No valid history rows in training split")
        classifier = _classifier(X["training"][train], need["training"][train],
                                 setting["seed"])
        active = train & need["training"] & valid["training"] & (timing["training"] >= 0)
        if not active.any():
            report[name] = {"status": "insufficient_feasible_training_burns",
                            "training_decisions": int(train.sum()),
                            "training_triggered": int((train & need["training"]).sum()),
                            "training_feasible_triggered": 0}
            continue
        regressors = []
        if active.sum() < 5:
            # A constant baseline remains meaningful for very small test fixtures.
            from sklearn.dummy import DummyRegressor
            for j in range(3):
                regressors.append(DummyRegressor().fit(X["training"][active], burn["training"][active, j]))
        else:
            for j in range(3):
                regressors.append(HistGradientBoostingRegressor(
                    max_iter=100, max_leaf_nodes=15, min_samples_leaf=5,
                    random_state=setting["seed"]).fit(X["training"][active],
                                                     burn["training"][active, j]))
        phase_model = _classifier(X["training"][active], timing["training"][active],
                                  setting["seed"])
        report[name] = {"status": "evaluated",
                        "training_decisions": int(train.sum()),
                        "training_triggered": int((train & need["training"]).sum()),
                        "training_feasible_triggered": int(active.sum())}
        for split in ("validation", "test"):
            rows = mask[split]
            y = need[split][rows]
            if not rows.any():
                raise ValueError(f"No usable {split} history rows")
            predicted = classifier.predict(X[split][rows]).astype(bool)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, predicted, average="binary", zero_division=0)
            feasible = rows & need[split] & valid[split] & (timing[split] >= 0)
            if feasible.any():
                pred_dv = np.column_stack([reg.predict(X[split][feasible]) for reg in regressors])
                rmse = float(np.sqrt(np.mean((pred_dv - burn[split][feasible]) ** 2)))
                phase_accuracy = float(np.mean(phase_model.predict(X[split][feasible]) == timing[split][feasible]))
            else:
                rmse, phase_accuracy = None, None
            report[name][split] = {
                "evaluated_decisions": int(rows.sum()),
                "positive_rate": float(np.mean(y)),
                "maneuver_precision": float(precision),
                "maneuver_recall": float(recall),
                "maneuver_f1": float(f1),
                "confusion_matrix_true_rows": confusion_matrix(y, predicted, labels=[False, True]).tolist(),
                "feasible_burn_examples": int(feasible.sum()),
                "delta_v_rmse_m_s": rmse,
                "burn_delay_class_accuracy_teacher_forced": phase_accuracy,
            }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
