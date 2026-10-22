"""Does more history help the 24 h-ahead model? Learning curves and a baseline comparison on fixed weeks.

Run in a notebook cell with `%run -m src.experiment` to see the plots inline; they are also saved to reports/.
Model settings are never changed here and the test weeks are only scored, never used to choose anything.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import PROCESSED, REPORTS, TEST_START, VAL_START
from .model import design, fit_ensemble, predict, time_split

BASELINE_FROM = "2021-08-09"  # the current model: Aug-Sep 2021 history only
HISTORY = ["2021-09-01", BASELINE_FROM, "2021-06-01", "2021-03-01", "2020-12-01", "2020-08-09", None]  # None = all
FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:  # too few ships to score this group
        return {"n": len(y), "rate": float(y.mean()) if len(y) else np.nan, "roc_auc": np.nan, "pr_auc": np.nan,
                "brier": np.nan, "cal_gap": np.nan}
    obs, pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    return {"n": len(y), "rate": y.mean(), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "cal_gap": float(np.abs(obs - pred).mean())}


def oob_roc(bundle, X, y):
    """Each member scored on the training rows its bootstrap left out (same draw as fit_ensemble)."""
    draws = np.random.default_rng(0).integers(0, len(X), (len(bundle["members"]), len(X)))
    scores = []
    for m, d in zip(bundle["members"], draws):
        out = np.setdiff1d(np.arange(len(X)), d)
        scores.append(roc_auc_score(y.iloc[out], m.predict_proba(X.iloc[out])[:, 1]))
    return float(np.mean(scores))


def run_one(df, y, X, train, val, test, new):
    b = fit_ensemble(X[train], y[train], X[val], y[val])
    p = predict(b, df)[0]
    raw = np.mean([m.predict_proba(X[train])[:, 1] for m in b["members"]], axis=0)
    return {"train_rows": int(train.sum()), "train_from": str(df.arrival[train].min().date()),
            "in_bag_roc": roc_auc_score(y[train], raw), "oob_roc": oob_roc(b, X[train], y[train]),
            "val": metrics(y[val], p[val]), "test": metrics(y[test], p[test]),
            "test_new_vessels": metrics(y[test & new], p[test & new])}, p


def paired_bootstrap(y, p_new, p_old, n=1000, seed=0):
    """95% interval for (more history - baseline) on the same test ships."""
    rng, y = np.random.default_rng(seed), np.asarray(y)
    d = {"roc_auc": [], "pr_auc": [], "brier": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d["roc_auc"].append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
        d["pr_auc"].append(average_precision_score(y[i], p_new[i]) - average_precision_score(y[i], p_old[i]))
        d["brier"].append(brier_score_loss(y[i], p_new[i]) - brier_score_loss(y[i], p_old[i]))
    return {k: [round(float(np.percentile(v, q)), 3) for q in (2.5, 50, 97.5)] for k, v in d.items()}


def main():
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED / "features.parquet").sort_values("arrival", ignore_index=True)
    calls = pd.read_parquet(PROCESSED / "calls.parquet")
    y, X = df.congested.astype(int), design(df)
    _, val, test = time_split(df)
    # "new vessel" = never seen at any of the ports before the test weeks, so it is new to every model
    seen = set(calls.mmsi[calls.arrival < pd.Timestamp(TEST_START)])
    new = ~df.mmsi.isin(seen).to_numpy()
    print(f"validation {VAL_START}..{TEST_START}: {val.sum()} ships | test from {TEST_START}: {test.sum()} ships, "
          f"{(test & new).sum()} new vessels | periods: {sorted(df.period.unique())}")

    history, preds = [], {}
    for start in HISTORY:
        train = time_split(df, start)[0]
        if history and train.sum() == history[-1]["train_rows"]:
            continue  # no extra data before this date
        r, p = run_one(df, y, X, train, val, test, new)
        r["label"] = "baseline (current)" if start == BASELINE_FROM else ("all history" if start is None else start)
        history.append(r)
        preds[r["label"]] = p
        print(f"{r['label']:>20} from {r['train_from']}: {r['train_rows']:>6} rows | ROC in-bag {r['in_bag_roc']:.3f} "
              f"OOB {r['oob_roc']:.3f} val {r['val']['roc_auc']:.3f} test {r['test']['roc_auc']:.3f} | "
              f"test Brier {r['test']['brier']:.3f} | new-vessel ROC {r['test_new_vessels']['roc_auc']:.3f}", flush=True)

    # same history, fewer rows: separates "more rows" from "older rows"
    full_train = time_split(df)[0]
    rng, fractions = np.random.default_rng(1), []
    for f in FRACTIONS:
        train = full_train & (rng.random(len(df)) < f) if f < 1 else full_train
        r, _ = run_one(df, y, X, train, val, test, new)
        r["fraction"] = f
        fractions.append(r)
        print(f"{f:>5.0%} of all history: {r['train_rows']:>6} rows | val ROC {r['val']['roc_auc']:.3f} "
              f"test ROC {r['test']['roc_auc']:.3f}", flush=True)

    # chosen on validation only; the test weeks never influence the choice
    chosen = max(history, key=lambda r: (round(r["val"]["roc_auc"], 3), r["train_rows"]))
    base = next(r for r in history if r["label"] == "baseline (current)")
    full = history[-1]
    yt = y[test].to_numpy()
    ci = paired_bootstrap(yt, preds[full["label"]][test], preds[base["label"]][test])
    ci_new = paired_bootstrap(y[test & new].to_numpy(), preds[full["label"]][test & new], preds[base["label"]][test & new])
    summary = {"validation picks": chosen["label"], "chosen train_from": chosen["train_from"],
               "all history - baseline, test (2.5%, median, 97.5%)": ci,
               "all history - baseline, new vessels": ci_new}
    print("\nchosen on validation:", chosen["label"], "->", f"python -m src.train --train-from {chosen['train_from']}")
    print("difference, all history minus baseline (95% interval):", json.dumps(summary, indent=1)[1:-1])

    per_port = {lbl: {port: metrics(y[test & (df.port == port).to_numpy()], preds[lbl][test & (df.port == port).to_numpy()])["roc_auc"]
                      for port in sorted(df.port.unique())} for lbl in (base["label"], full["label"])}
    out = {"history_curve": history, "fraction_curve": fractions, "summary": summary, "test_roc_by_port": per_port}
    (REPORTS / "experiment.json").write_text(json.dumps(out, indent=2, default=float))

    rows = [r["train_rows"] for r in history]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0, 0]
    for key, name in [("in_bag_roc", "train (in-bag)"), ("oob_roc", "train (out-of-bag)")]:
        a.plot(rows, [r[key] for r in history], "o--", label=name)
    for split in ["val", "test"]:
        a.plot(rows, [r[split]["roc_auc"] for r in history], "o-", label=split)
    a.plot(rows, [r["test_new_vessels"]["roc_auc"] for r in history], "s:", label="test, new vessels")
    for r in history:
        a.annotate(r["label"], (r["train_rows"], r["test"]["roc_auc"]), fontsize=7, rotation=20)
    a.set(xscale="log", xlabel="training rows (more history to the right)", ylabel="ROC-AUC", title="Learning curve: history length")
    a.legend(fontsize=8)
    a = ax[0, 1]
    for split in ["val", "test"]:
        a.plot(rows, [r[split]["brier"] for r in history], "o-", label=f"Brier, {split}")
        a.plot(rows, [r[split]["cal_gap"] for r in history], "o--", label=f"calibration gap, {split}")
    a.set(xscale="log", xlabel="training rows", ylabel="lower is better", title="Brier score and calibration gap")
    a.legend(fontsize=8)
    a = ax[1, 0]
    fr = [r["train_rows"] for r in fractions]
    a.plot(fr, [r["oob_roc"] for r in fractions], "o--", label="train (out-of-bag)")
    for split in ["val", "test"]:
        a.plot(fr, [r[split]["roc_auc"] for r in fractions], "o-", label=split)
    a.set(xscale="log", xlabel="training rows (random share of all history)", ylabel="ROC-AUC",
          title="Learning curve: same history, fewer rows")
    a.legend(fontsize=8)
    a = ax[1, 1]
    for lbl in (base["label"], full["label"]):
        obs, pred = calibration_curve(yt, preds[lbl][test], n_bins=8, strategy="quantile")
        a.plot(pred, obs, "o-", label=lbl)
    a.plot([0, 1], [0, 1], "--", color="grey")
    a.set(xlabel="predicted probability", ylabel="observed share congested", title="Calibration on the test weeks")
    a.legend(fontsize=8)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):  # plain row counts on the log axes
        a.xaxis.set_major_formatter(ScalarFormatter())
        a.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "experiment_curves.png", dpi=150)
    plt.show()

    fig, a = plt.subplots(figsize=(9, 4))
    ports = list(per_port[base["label"]])
    xs = np.arange(len(ports))
    for k, lbl in enumerate((base["label"], full["label"])):
        a.bar(xs + (k - 0.5) * 0.38, [per_port[lbl][p] for p in ports], 0.38, label=lbl)
    a.axhline(0.5, color="grey", ls="--")
    a.set(xticks=xs, xticklabels=ports, ylabel="test ROC-AUC", title="Test ROC-AUC by port", ylim=(0.4, 0.9))
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "experiment_ports.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
