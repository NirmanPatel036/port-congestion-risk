"""Choose the High / Medium / Low cut-offs on member disagreement, using the validation weeks only.

`python -m src.confidence` re-tunes an already trained model in place: the 5 members and the calibrator are
untouched, only `confidence_cuts` in the bundle changes. `src.train` calls `tune` itself on new models.
"""
import argparse
import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .config import LEAD_H, MODELS, PROCESSED, REPORTS, tag
from .model import DEFAULT_CUTS, confidence, predict, time_split

MIN_SHARE = {"High": 0.30, "Medium": 0.15, "Low": 0.10}   # every level must cover enough ships to mean something
MAX_SHARE = {"High": 0.60, "Medium": 0.60, "Low": 0.30}
BOOTSTRAPS = 200


def table(y, prob, level):
    """Accuracy (at 50%), Brier score and coverage for each confidence level."""
    y, prob, level = np.asarray(y), np.asarray(prob), np.asarray(level)
    rows = {}
    for lv in ["High", "Medium", "Low"]:
        m = level == lv
        rows[lv] = {"share": round(float(m.mean()), 3), "n": int(m.sum()),
                    "accuracy": round(float(((prob[m] >= 0.5) == y[m]).mean()), 3) if m.any() else None,
                    "brier": round(float(brier_score_loss(y[m], prob[m])), 3) if m.any() else None,
                    "congested": round(float(y[m].mean()), 3) if m.any() else None}
    return rows


def separation(correct, level):
    """Smallest step in accuracy between neighbouring levels; both steps must be positive to be useful."""
    acc = [correct[level == lv].mean() for lv in ["High", "Medium", "Low"]]
    return min(acc[0] - acc[1], acc[1] - acc[2])


def tune(y, prob, spread):
    """Pick (high, low) disagreement cut-offs that keep coverage sensible and separate the levels most reliably.
    Candidates are quantiles of the validation spread; each is scored on bootstrap resamples so a lucky split
    of a few ships does not win."""
    y, prob, spread = np.asarray(y), np.asarray(prob), np.asarray(spread)
    correct = (prob >= 0.5) == y
    cands = np.unique(np.round(np.quantile(spread, np.arange(0.05, 0.96, 0.05)), 4))
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAPS)]
    best = None
    for hi, lo in itertools.combinations(cands, 2):
        level = confidence(spread, (hi, lo))
        share = {lv: (level == lv).mean() for lv in MIN_SHARE}
        if any(share[lv] < MIN_SHARE[lv] or share[lv] > MAX_SHARE[lv] for lv in share):
            continue
        brier = [brier_score_loss(y[level == lv], prob[level == lv]) for lv in ["High", "Medium", "Low"]]
        if not (brier[0] < brier[1] < brier[2] and separation(correct, level) > 0):
            continue
        score = np.mean([separation(correct[i], level[i]) for i in boots])
        if best is None or score > best[0]:
            best = (score, (float(hi), float(lo)))
    if best is None:
        print("no cut-offs meet the coverage and ordering rules; keeping", DEFAULT_CUTS)
        return DEFAULT_CUTS, None
    return best[1], round(float(best[0]), 3)


def main():
    ap = argparse.ArgumentParser(description="Re-tune confidence cut-offs of a trained model on validation")
    ap.add_argument("--lead-h", type=int, default=LEAD_H)
    sfx = tag(ap.parse_args().lead_h)
    path = MODELS / f"model{sfx}.joblib"
    bundle = joblib.load(path)
    df = pd.read_parquet(PROCESSED / f"features{sfx}.parquet").sort_values("arrival", ignore_index=True)
    _, val, test = time_split(df)
    prob, spread, old = predict(bundle, df)
    y = df.congested.astype(int).to_numpy()
    old_cuts = tuple(bundle.get("confidence_cuts", DEFAULT_CUTS))

    cuts, score = tune(y[val], prob[val], spread[val])
    new = confidence(spread, cuts)
    out = {"old cuts (High below, Low above)": old_cuts, "new cuts (High below, Low above)": cuts,
           "validation, bootstrap mean of smallest accuracy step": score,
           "validation, old cuts": table(y[val], prob[val], old[val]),
           "validation, new cuts": table(y[val], prob[val], new[val]),
           # scored once after the cut-offs were fixed; never used to choose them
           "test check, new cuts": table(y[test], prob[test], new[test])}
    bundle["confidence_cuts"] = cuts
    joblib.dump(bundle, path)
    (REPORTS / f"confidence{sfx}.json").write_text(json.dumps(out, indent=2))
    print(f"{path.name}: cut-offs {old_cuts} -> {cuts}")
    for name in ["validation, old cuts", "validation, new cuts", "test check, new cuts"]:
        print(f"\n{name}\n", pd.DataFrame(out[name]).T.to_string())


if __name__ == "__main__":
    main()
