import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import DELAY_H, MODELS, PROCESSED, REPORTS
from src.explain import explain
from src.features import backlog_days
from src.model import contributions, predict

st.set_page_config(page_title="Port Congestion Risk", layout="wide")


@st.cache_resource
def load():
    return joblib.load(MODELS / "model.joblib"), pd.read_parquet(PROCESSED / "features.parquet")


bundle, df = load()
st.title("Port congestion risk")
st.caption(f"Chance that an arriving ship waits more than {DELAY_H} h for a berth, why, and what to do about it.")

port = st.sidebar.selectbox("Port", sorted(df.port.unique()))
metrics_file = REPORTS / "metrics.json"
if metrics_file.exists():
    m = json.loads(metrics_file.read_text())["xgboost ensemble (calibrated)"]
    st.sidebar.caption(f"Test ROC-AUC {m['roc_auc']} | PR-AUC {m['pr_auc']} | Brier {m.get('brier')}")

p = df[df.port == port].sort_values("arrival")
st.subheader(f"Ships waiting at {port}")
st.line_chart(p.set_index("arrival")["waiting_now"], height=220)

test = p[p.arrival >= bundle["test_start"]]
if test.empty:
    st.info("No arrivals for this port in the held-out test period.")
    st.stop()

st.subheader("Score an arriving ship (held-out period)")
i = st.selectbox("Arrival", test.index[::-1],
                 format_func=lambda i: f"{test.arrival[i]:%Y-%m-%d %H:%M} | MMSI {test.mmsi[i]} | {test.length_m[i]:.0f} m")
row = test.loc[[i]].copy()
base = row.iloc[0]

with st.expander("What if conditions were different?"):
    row["waiting_now"] = st.slider("Ships waiting", 0, int(df.waiting_now.max()), int(base.waiting_now))
    row["gust_max_24h_kn"] = st.slider("Max gust in next 24 h (kn)", 0, 60, int(np.nan_to_num(base.gust_max_24h_kn)))
    row["recent_wait_h"] = st.slider("Median wait of ships berthed this week (h)", 0,
                                     int(np.nanmax(df.recent_wait_h)), int(np.nan_to_num(base.recent_wait_h)))
    row["backlog_days"] = backlog_days(row.waiting_now, row.berths_7d)

prob, spread, conf = predict(bundle, row)
c1, c2, c3 = st.columns(3)
c1.metric(f"Chance of waiting > {DELAY_H} h", f"{prob[0]:.0%}")
c2.metric("Confidence", conf[0], help=f"Spread between the 5 ensemble models: {spread[0]:.2f}")
outcome = f"{base.wait_h:.0f} h wait" + (" (still waiting at data end)" if base.censored else "")
c3.metric("What actually happened", outcome)

raising, lowering, actions = explain(row.iloc[0], contributions(bundle, row).iloc[0])
left, right = st.columns(2)
with left:
    st.markdown("**Why**")
    for _, why, _ in raising:
        st.markdown(f"- {why}")
    if not raising:
        st.markdown("- No strong risk drivers")
    if lowering:
        st.markdown("**Working in its favour**")
        for why in lowering:
            st.markdown(f"- {why}")
with right:
    st.markdown("**Recommended actions**")
    for a in actions or ["Plan normally; no action needed."]:
        st.markdown(f"- {a}")
