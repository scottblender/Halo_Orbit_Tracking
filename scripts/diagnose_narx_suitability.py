"""Reproducible held-out diagnostics for the NARX-inspired station-keeping policy.

Runs on the existing teacher_training/{training,validation,test}.npz files.
Does not regenerate trajectories or claim to establish closed-loop suitability.
Usage: python scripts/diagnose_narx_suitability.py
"""
from pathlib import Path
import json
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from halo_tracking.two_stage_narx import _features, _timing_classes
from halo_tracking.study import load_config


def load_split(path, cfg):
    with np.load(path, allow_pickle=False) as raw:
        d = {k: raw[k] for k in raw.files}
    x, history_ok = _features(d, "observer")
    needed = d["observer_maneuver_required"].ravel().astype(bool)
    valid = d["observer_maneuver_valid"].ravel().astype(bool)
    dv = d["observer_target_delta_v_rotating_m_s"].reshape(-1, 3)
    period = float(d["decision_times_tu"][-1]) / cfg["narx"]["arc_duration_periods"]
    timing = _timing_classes(d["observer_target_burn_delay_tu"].ravel(),
                             cfg["narx"]["candidate_delay_fractions"], period)
    mask = history_ok & np.isfinite(x).all(axis=1)
    feasible = mask & needed & valid & (timing >= 0) & np.isfinite(dv).all(axis=1)
    return dict(x=x, mask=mask, y=needed, valid=valid, dv=dv,
                timing=timing, feasible=feasible)


def score_classification(y, predicted):
    p, r, f, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0)
    cm = confusion_matrix(y, predicted, labels=[False, True])
    return dict(precision=float(p), recall=float(r), f1=float(f),
                tn=int(cm[0, 0]), fp=int(cm[0, 1]),
                fn=int(cm[1, 0]), tp=int(cm[1, 1]))


def make_models(label, seed):
    if label == "dummy":
        return DummyClassifier(strategy="most_frequent"), DummyRegressor(strategy="mean")
    if label == "linear":
        return LogisticRegression(max_iter=2000, class_weight="balanced"), Ridge(alpha=10.)
    if label == "decision_tree":
        return (DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=seed),
                DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed))
    if label == "narx_boosted":
        return (HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                min_samples_leaf=5, random_state=seed),
                HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                               min_samples_leaf=5, random_state=seed))
    raise ValueError(label)


def evaluate(data, train, name, seed):
    xtr = train["x"][train["mask"]]
    ytr = train["y"][train["mask"]]
    burntr = train["x"][train["feasible"]]
    targettr = train["dv"][train["feasible"]]
    classifier, reg_template = make_models(name, seed)
    if np.unique(ytr).size < 2:
        classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(xtr, ytr)
    regressors = []
    for i in range(3):
        _, reg = make_models(name, seed)
        reg.fit(burntr, targettr[:, i])
        regressors.append(reg)
    scores = {}
    for split in ("validation", "test"):
        d = data[split]
        x = d["x"][d["mask"]]
        actual = d["y"][d["mask"]]
        predicted = classifier.predict(x)
        classification = score_classification(actual, predicted)
        classification["n_decisions"] = int(actual.size)
        classification["triggered_but_infeasible"] = int(
            np.sum(d["mask"] & d["y"] & ~d["feasible"]))
        selected = d["feasible"]
        if np.any(selected):
            v = np.column_stack([model.predict(d["x"][selected]) for model in regressors])
            e = v - d["dv"][selected]
            classification["teacher_forced_burn_rmse_m_s"] = float(np.sqrt(np.mean(e**2)))
            classification["teacher_forced_burn_vector_rmse_m_s"] = float(
                np.sqrt(np.mean(np.sum(e**2, axis=1))))
            classification["feasible_burn_count"] = int(selected.sum())
        scores[split] = classification
    return scores


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config()
    path = root / "results/one_year/teacher_training"
    if not all((path / (name + ".npz")).exists()
               for name in ("training", "validation", "test")):
        raise FileNotFoundError(
            "Run the Notebook 01 teacher-data generation first: " + str(path))
    data = {k: load_split(path / (k + ".npz"), cfg)
            for k in ("training", "validation", "test")}
    if not np.any(data["training"]["feasible"]):
        raise ValueError("No feasible burn targets on training split")
    report = {
        "scope": "held-out offline imitation only; NOT closed-loop control performance",
        "split_unit": "independent trajectory arcs, not shuffled decision rows",
        "label_semantics": "classifier predicts maneuver_required; regressors train/evaluate only on feasible valid teacher burns",
        "models": {name: evaluate(data, data["training"], name, cfg["narx"]["seed"])
                   for name in ("dummy", "linear", "decision_tree", "narx_boosted")},
    }
    out = root / "results/one_year/narx_suitability"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    names = list(report["models"])
    test = [report["models"][n]["test"] for n in names]
    axes[0].bar(names, [v["recall"] for v in test])
    axes[0].set_title("Held-out maneuver recall")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(names, [v["teacher_forced_burn_vector_rmse_m_s"] for v in test])
    axes[1].set_title("Teacher-forced burn vector RMSE (m/s)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(out / "offline_baselines.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print("Written:", out / "diagnostics.json")


if __name__ == "__main__":
    main()
