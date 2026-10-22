import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DELAY_H, LEAD_H, MODELS, PROCESSED, REPORTS, TEST_START, tag
from .model import DEVICE, contributions, design, fit_ensemble, fit_xgb, predict, time_split


def scores(y, score, pred):
    out = {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score), "f1": f1_score(y, pred)}
    if score.min() >= 0 and score.max() <= 1:
        out["brier"] = brier_score_loss(y, score)
    return {k: round(float(v), 3) for k, v in out.items()}


def calibration(y, prob, bins=8):
    """Predicted vs observed rate in equal-size bins; the mean gap is the expected calibration error."""
    observed, predicted = calibration_curve(y, prob, n_bins=bins, strategy="quantile")
    return {"predicted": predicted.round(3).tolist(), "observed": observed.round(3).tolist(),
            "mean |gap|": round(float(np.abs(observed - predicted).mean()), 3)}


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate on features built LEAD hours before arrival")
    ap.add_argument("--lead-h", type=int, default=LEAD_H, help="0 = at arrival (baseline)")
    ap.add_argument("--train-from", help="first arrival date used for training (default: all history)")
    args = ap.parse_args()
    lead_h = args.lead_h
    sfx = tag(lead_h)
    MODELS.mkdir(exist_ok=True)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED / f"features{sfx}.parquet").sort_values("arrival", ignore_index=True)
    y = df.congested.astype(int)
    X = design(df)

    # split by fixed dates, never at random: the model only sees the past, like it would in production
    fit, cal, test = time_split(df, args.train_from)
    print(f"device={DEVICE} fit={fit.sum()} calibrate={cal.sum()} test={test.sum()} congested={y.mean():.0%}")

    bundle = fit_ensemble(X[fit], y[fit], X[cal], y[cal])
    bundle["test_start"], bundle["lead_h"] = pd.Timestamp(TEST_START), lead_h
    bundle["train_from"] = args.train_from or str(df.arrival[fit].min().date())
    prob, spread, conf = predict(bundle, df[test])

    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    logreg.fit(X[fit | cal], y[fit | cal])
    lr_prob = logreg.predict_proba(X[test])[:, 1]
    backlog = df.backlog_days[test].to_numpy()

    yt = y[test].to_numpy()
    results = {
        "prediction lead (h before arrival)": lead_h,
        f"base rate (waits > {DELAY_H} h)": round(float(yt.mean()), 3),
        "rule: backlog > 1 day": scores(yt, backlog, backlog > 1),
        "logistic regression": scores(yt, lr_prob, lr_prob >= 0.5),
        "xgboost ensemble (calibrated)": scores(yt, prob, prob >= 0.5),
        "confidence mix on test": pd.Series(conf).value_counts(normalize=True).round(2).to_dict(),
        "accuracy by confidence": {c: round(float(((prob >= 0.5) == yt)[conf == c].mean()), 3) for c in np.unique(conf)},
        "calibration (test, 8 equal-size bins)": calibration(yt, prob),
    }

    per_port = {}
    for port, g in df[test].groupby("port"):
        p = prob[(df[test].port == port).to_numpy()]
        if g.congested.nunique() == 2:
            per_port[port] = {"n": len(g), "congested": round(float(g.congested.mean()), 2),
                              "roc_auc": round(float(roc_auc_score(g.congested, p)), 3)}
    results["xgboost by port (test)"] = per_port

    # can a model trained on four ports judge a port it has never seen?
    Xp, lopo = design(df, with_port=False), {}
    for port in df.port.unique():
        held = df.port == port
        if y[held].nunique() == 2:
            m = fit_xgb(Xp[~held], y[~held])
            lopo[port] = round(float(roc_auc_score(y[held], m.predict_proba(Xp[held])[:, 1])), 3)
    results["leave-one-port-out roc_auc"] = lopo

    joblib.dump(bundle, MODELS / f"model{sfx}.joblib")
    (REPORTS / f"metrics{sfx}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    cal_bins = results["calibration (test, 8 equal-size bins)"]
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.plot(cal_bins["predicted"], cal_bins["observed"], "o-")
    plt.xlabel("predicted probability")
    plt.ylabel("observed share congested")
    plt.title(f"Calibration (test period, {lead_h} h ahead)")
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / f"calibration{sfx}.png", dpi=150)

    imp = contributions(bundle, df[test]).abs().mean().sort_values()
    plt.figure(figsize=(6, 5))
    imp.plot.barh()
    plt.xlabel("mean |contribution| to log-odds")
    plt.title(f"What drives the predictions ({lead_h} h ahead)")
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / f"drivers{sfx}.png", dpi=150)


if __name__ == "__main__":
    main()
