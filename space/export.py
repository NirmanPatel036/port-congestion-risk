import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import app
from src.model import predict

F32 = np.float32
np.seterr(over="ignore")  # 1e308 -> inf in float32 is intended


def flat_tree(node_json, columns):
    """XGBoost JSON dump -> flat arrays: feature index (-1 = leaf), threshold, yes, no, missing, leaf value."""
    nodes = {}

    def walk(n):
        nodes[n["nodeid"]] = n
        for c in n.get("children", []):
            walk(c)

    # XGBoost can write split thresholds of +-inf, which JSON does not allow; 1e308 rounds to inf in float32
    walk(json.loads(re.sub(r":\s*(-?)inf\b", r": \g<1>1e308", node_json)))
    order = sorted(nodes)
    idx = {nid: k for k, nid in enumerate(order)}
    out = {"f": [], "t": [], "y": [], "n": [], "m": [], "v": []}
    for nid in order:
        n = nodes[nid]
        leaf = "leaf" in n
        out["f"].append(-1 if leaf else columns.index(n["split"]))
        out["t"].append(0 if leaf else float(n["split_condition"]))
        out["y"].append(0 if leaf else idx[n["yes"]])
        out["n"].append(0 if leaf else idx[n["no"]])
        out["m"].append(0 if leaf else idx[n["missing"]])
        out["v"].append(float(n["leaf"]) if leaf else 0)
    return out


def tree_value(tree, x):
    k = 0
    while tree["f"][k] >= 0:
        v = x[tree["f"][k]]
        if np.isnan(v):
            k = tree["m"][k]
        else:
            k = tree["y"][k] if F32(v) < F32(tree["t"][k]) else tree["n"][k]
    return tree["v"][k]


def export_model(bundle):
    columns = bundle["columns"]
    members = []
    for m in bundle["members"]:
        bst = m.get_booster()
        base = float(json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
        members.append({"base_margin": float(np.log(base / (1 - base))),
                        "trees": [flat_tree(t, columns) for t in bst.get_dump(dump_format="json")]})
    cal = bundle["calibrator"]
    return {"columns": columns, "members": members, "cal_coef": float(cal.coef_[0][0]),
            "cal_intercept": float(cal.intercept_[0]), "cuts": list(bundle.get("confidence_cuts", (0.05, 0.12)))}


def vector(row, columns):
    return np.array([(1.0 if row["port"] == c[5:] else 0.0) if c.startswith("port_") else
                     (np.nan if pd.isna(row[c]) else float(row[c])) for c in columns])


def replay(model, row):
    """Same maths the page runs in JavaScript."""
    x = vector(row, model["columns"])
    probs = []
    for mem in model["members"]:
        margin = mem["base_margin"] + sum(tree_value(t, x) for t in mem["trees"])
        probs.append(1 / (1 + np.exp(-margin)))
    avg = np.clip(np.mean(probs), 1e-4, 1 - 1e-4)
    z = model["cal_coef"] * np.log(avg / (1 - avg)) + model["cal_intercept"]
    return 1 / (1 + np.exp(-z)), float(np.std(probs))


def ships():
    out = []
    for port in app.PORTS:
        for label, i in app.ship_choices(port, app.SHOW[0]):
            base, prob, conf, spread, raising, lowering, actions = app.score(i)
            r = app.TEST.loc[i]
            t = app.local_time(r.arrival, port)
            num = lambda v: None if pd.isna(v) else float(v)
            out.append({
                "id": int(i), "port": port, "label": label, "congested": int(r.congested), "censored": bool(r.censored),
                "wait_h": num(r.wait_h), "due": f"{t:%a %d %b, %H:%M} {t.tzname()}", "t": int(r.arrival.timestamp() * 1000),
                "x": {c: (r[c] if c == "port" else num(r[c])) for c in ["port"] + [f for f in app.DF.columns
                      if f in {"waiting_now", "backlog_days", "arrivals_24h", "berths_7d", "at_berth_now", "recent_wait_h",
                               "gust_kn", "gust_max_prev_24h_kn", "precip_prev_24h_mm", "hour_local", "is_weekend",
                               "length_m", "draft_m", "is_tanker"}]},
                "prob": prob, "conf": conf, "spread": spread,
                "up": [why for _, why, _ in raising], "down": list(lowering), "actions": list(actions)})
    return out


def main():
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "space/static")
    dest.mkdir(parents=True, exist_ok=True)
    model = export_model(app.BUNDLE)
    data = ships()

    # the tree copy must reproduce the real model, or the sliders would disagree with the scores
    worst = 0.0
    for s in data:
        row = dict(s["x"])
        p, spread = replay(model, row)
        worst = max(worst, abs(p - s["prob"]), abs(spread - s["spread"]))
    print(f"tree copy vs XGBoost on {len(data)} ships: largest difference {worst:.2e}")
    assert worst < 1e-4, "exported trees do not match the model"

    queue = {p: [[int(t.timestamp() * 1000), int(w)] for t, w in app.queue_history(p).itertuples(index=False)]
             for p in app.PORTS}
    meta = {"lead_h": app.LEAD_H, "delay_h": app.DELAY_H, "n_test": len(app.TEST),
            "roc_auc": app.METRICS.get("roc_auc"), "max_waiting": int(app.DF.waiting_now.max()),
            "max_recent": int(app.DF.recent_wait_h.max())}
    (dest / "data.json").write_text(json.dumps({"meta": meta, "ships": data, "queue": queue}, separators=(",", ":")))
    (dest / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    for f in ["data.json", "model.json"]:
        print(f"{f}: {(dest / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
