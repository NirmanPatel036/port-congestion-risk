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

from .config import DELAY_H, MODELS, PROCESSED, REPORTS
from .model import DEVICE, contributions, design, fit_ensemble, fit_xgb, predict


def scores(y, score, pred):
    out = {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score), "f1": f1_score(y, pred)}
    if score.min() >= 0 and score.max() <= 1:
        out["brier"] = brier_score_loss(y, score)
    return {k: round(float(v), 3) for k, v in out.items()}


def main():
    MODELS.mkdir(exist_ok=True)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED / "features.parquet").sort_values("arrival", ignore_index=True)
    y = df.congested.astype(int)
    X = design(df)

    # split by time, never at random: the model only sees the past, like it would in production
    test_start = df.arrival.quantile(0.75)
    cal_start = df.arrival[df.arrival < test_start].quantile(0.8)
    fit, cal, test = df.arrival < cal_start, (df.arrival >= cal_start) & (df.arrival < test_start), df.arrival >= test_start
    print(f"device={DEVICE} fit={fit.sum()} calibrate={cal.sum()} test={test.sum()} congested={y.mean():.0%}")

    bundle = fit_ensemble(X[fit], y[fit], X[cal], y[cal])
    bundle["test_start"] = test_start
    prob, spread, conf = predict(bundle, df[test])

    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    logreg.fit(X[~test], y[~test])
    lr_prob = logreg.predict_proba(X[test])[:, 1]
    backlog = df.backlog_days[test].to_numpy()

    yt = y[test].to_numpy()
    results = {
        f"base rate (waits > {DELAY_H} h)": round(float(yt.mean()), 3),
        "rule: backlog > 1 day": scores(yt, backlog, backlog > 1),
        "logistic regression": scores(yt, lr_prob, lr_prob >= 0.5),
        "xgboost ensemble (calibrated)": scores(yt, prob, prob >= 0.5),
        "confidence mix on test": pd.Series(conf).value_counts(normalize=True).round(2).to_dict(),
        "accuracy by confidence": {c: round(float(((prob >= 0.5) == yt)[conf == c].mean()), 3) for c in np.unique(conf)},
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

    joblib.dump(bundle, MODELS / "model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    frac, mean_p = calibration_curve(yt, prob, n_bins=8, strategy="quantile")
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.plot(mean_p, frac, "o-")
    plt.xlabel("predicted probability")
    plt.ylabel("observed share congested")
    plt.title("Calibration (test period)")
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / "calibration.png", dpi=150)

    imp = contributions(bundle, df[test]).abs().mean().sort_values()
    plt.figure(figsize=(6, 5))
    imp.plot.barh()
    plt.xlabel("mean |contribution| to log-odds")
    plt.title("What drives the predictions")
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / "drivers.png", dpi=150)


if __name__ == "__main__":
    main()
