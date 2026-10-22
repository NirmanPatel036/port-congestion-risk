import html
import json

import gradio as gr
import joblib
import pandas as pd

from src.config import DELAY_H, MODELS, PROCESSED, REPORTS, TZ
from src.explain import explain
from src.features import backlog_days
from src.model import contributions, predict

BUNDLE = joblib.load(MODELS / "model.joblib")
DF = pd.read_parquet(PROCESSED / "features.parquet").sort_values("arrival", ignore_index=True)
LEAD_H = BUNDLE.get("lead_h", 24)
TEST = DF[DF.arrival >= BUNDLE["test_start"]]
PORTS = sorted(TEST.port.unique())
METRICS_FILE = REPORTS / "metrics.json"
METRICS = json.loads(METRICS_FILE.read_text())["xgboost ensemble (calibrated)"] if METRICS_FILE.exists() else {}
SHOW = ["All arrivals", f"Waited over {DELAY_H} h", f"Berthed within {DELAY_H} h"]


def local_time(t, port):
    return t.tz_localize("UTC").tz_convert(TZ[port])


def ship_label(r):
    t = local_time(r.arrival, r.port)
    kind = "tanker" if r.is_tanker else "cargo ship"
    return f"{t:%a %d %b, %H:%M} {t.tzname()}  ·  {r.length_m:.0f} m {kind}  ·  MMSI {int(r.mmsi)}"


def ship_choices(port, show):
    t = TEST[TEST.port == port]
    if show == SHOW[1]:
        t = t[t.congested == 1]
    elif show == SHOW[2]:
        t = t[t.congested == 0]
    return [(ship_label(r), int(i)) for i, r in t.iloc[::-1].iterrows()]  # newest first


def queue_history(port):
    p = DF[(DF.port == port) & (DF.arrival >= BUNDLE["test_start"] - pd.Timedelta(days=60))]
    return p[["arrival", "waiting_now"]].rename(columns={"arrival": "Arrival", "waiting_now": "Ships waiting"})


def score(i, waiting=None, gust=None, recent=None):
    row = TEST.loc[[i]].copy()
    for col, val in [("waiting_now", waiting), ("gust_max_prev_24h_kn", gust), ("recent_wait_h", recent)]:
        if val is not None:
            row[col] = float(val)
    row["backlog_days"] = backlog_days(row.waiting_now, row.berths_7d)
    prob, spread, conf = predict(BUNDLE, row)
    raising, lowering, actions = explain(row.iloc[0], contributions(BUNDLE, row).iloc[0])
    return row.iloc[0], float(prob[0]), conf[0], float(spread[0]), raising, lowering, actions


def risk_card(prob, conf, spread, base):
    band, word = ("high", "High risk") if prob >= 0.6 else ("mid", "Elevated risk") if prob >= 0.3 else ("low", "Low risk")
    waited = f"{base.wait_h:.0f} h" if pd.notna(base.wait_h) else "unknown"
    if base.censored:
        outcome = f"Still waiting after {waited} when the data ends"
    else:
        outcome = f"Actually waited {waited}" + (" — congested" if base.congested == 1 else " — on time")
    return f"""
    <div class="risk-card band-{band}">
      <div class="risk-top"><span class="risk-kicker">Chance of waiting more than {DELAY_H} h</span>
        <span class="risk-word">{word}</span></div>
      <div class="risk-value">{prob:.0%}</div>
      <div class="risk-bar"><span style="width:{prob * 100:.0f}%"></span></div>
      <div class="pills">
        <span class="pill conf-{conf.lower()}" title="Spread between the 5 ensemble models: {spread:.3f}">{conf} confidence</span>
        <span class="pill">{html.escape(outcome)}</span>
      </div>
    </div>"""


def conditions(base):
    t = local_time(base.arrival, base.port)
    tiles = [("Ships waiting", f"{base.waiting_now:.0f}"), ("Backlog", f"{base.backlog_days:.1f} days"),
             ("Recent median wait", "–" if pd.isna(base.recent_wait_h) else f"{base.recent_wait_h:.0f} h"),
             ("Max gust, past 24 h", "–" if pd.isna(base.gust_max_prev_24h_kn) else f"{base.gust_max_prev_24h_kn:.0f} kn")]
    cells = "".join(f'<div class="tile"><div class="tile-v">{v}</div><div class="tile-k">{k}</div></div>' for k, v in tiles)
    return (f'<div class="sub">Port conditions {LEAD_H} h before the ship is due ({t:%a %d %b, %H:%M} '
            f'{t.tzname()})</div><div class="tiles">{cells}</div>')


def reasons(raising, lowering):
    up = "".join(f'<li><span class="arrow up">▲</span>{html.escape(why)}</li>' for _, why, _ in raising)
    down = "".join(f'<li><span class="arrow down">▼</span>{html.escape(why)}</li>' for why in lowering)
    body = (up or '<li class="muted">No strong risk drivers</li>') + down
    return f'<div class="panel"><div class="panel-h">Why</div><ul class="list">{body}</ul></div>'


def advice(actions):
    items = "".join(f"<li>{html.escape(a)}</li>" for a in actions) or "<li>Plan normally; no action needed.</li>"
    return f'<div class="panel"><div class="panel-h">What to do now</div><ol class="list actions">{items}</ol></div>'


def render(i, waiting=None, gust=None, recent=None):
    if i is None:
        empty = '<div class="panel muted">No arrivals match this filter.</div>'
        return empty, "", "", ""
    base, prob, conf, spread, raising, lowering, actions = score(i, waiting, gust, recent)
    return risk_card(prob, conf, spread, base), conditions(TEST.loc[i]), reasons(raising, lowering), advice(actions)


def pick_port(port, show):
    choices = ship_choices(port, show)
    return gr.update(choices=choices, value=choices[0][1] if choices else None), queue_history(port)


def pick_ship(i):
    if i is None:
        return (*render(None), gr.update(), gr.update(), gr.update())
    b = TEST.loc[i]
    num = lambda x: 0 if pd.isna(x) else int(x)
    sliders = (gr.update(value=num(b.waiting_now)), gr.update(value=num(b.gust_max_prev_24h_kn)),
               gr.update(value=num(b.recent_wait_h)))
    return (*render(i), *sliders)


HEADER = f"""
<div class="hero">
  <div class="hero-kicker">Port congestion risk · {LEAD_H} h ahead</div>
  <h1>Will this ship wait for a berth?</h1>
  <p>Pick a US port and an arriving ship. The model reads the port as it stood {LEAD_H} hours before arrival,
  gives the chance of waiting more than {DELAY_H} hours, how sure it is, why, and what a planner can still do.</p>
  <div class="chips">
    <span>5 US ports</span><span>AIS Aug 2019 – Oct 2021</span><span>{len(TEST):,} unseen test arrivals</span>
    {f'<span>Test ROC-AUC {METRICS["roc_auc"]:.2f}</span>' if METRICS else ''}
  </div>
</div>"""

CSS = """
/* sea-style page: sky to shallow water, faint wave lines; deep ocean in dark mode */
:root {--body-background-fill: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='28' viewBox='0 0 160 28'%3E%3Cpath d='M0 14 Q 20 2 40 14 T 80 14 T 120 14 T 160 14' fill='none' stroke='%230ea5e9' stroke-opacity='0.10' stroke-width='2'/%3E%3C/svg%3E\"), linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 35%, #ccfbf1 100%) !important;}
:root.dark, :root .dark {--body-background-fill: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='28' viewBox='0 0 160 28'%3E%3Cpath d='M0 14 Q 20 2 40 14 T 80 14 T 120 14 T 160 14' fill='none' stroke='%230ea5e9' stroke-opacity='0.10' stroke-width='2'/%3E%3C/svg%3E\"), linear-gradient(180deg, #020617 0%, #0b2545 50%, #0f3d3e 100%) !important;}
body, gradio-app {background: var(--body-background-fill) !important; background-attachment: fixed !important;}
.gradio-container {max-width: 1120px !important; margin: auto; background: transparent !important;}
.hero {padding: 28px 30px; border-radius: 18px; color: #fff;
  background: radial-gradient(1200px 300px at 0% 0%, #14b8a6 0%, transparent 60%), linear-gradient(120deg, #0f172a, #134e4a);}
.hero h1 {font-size: 30px; margin: 6px 0 8px; color: #ffffff !important; letter-spacing: -0.02em;}
.hero p {color: #cffafe !important; max-width: 760px; margin: 0; font-size: 15px; line-height: 1.55;}
.hero-kicker {text-transform: uppercase; letter-spacing: .12em; font-size: 12px; color: #5eead4; font-weight: 600;}
.chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.chips span {background: #ccfbf1; color: #134e4a !important; border: 1px solid #5eead4; padding: 4px 12px;
  border-radius: 999px; font-size: 12px; font-weight: 600;}
.risk-card {padding: 22px 24px; border-radius: 16px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill);}
.risk-top {display: flex; justify-content: space-between; align-items: center; gap: 8px;}
.risk-kicker {font-size: 13px; color: var(--body-text-color-subdued);}
.risk-word {font-weight: 600; font-size: 13px; padding: 3px 10px; border-radius: 999px;}
.risk-value {font-size: 64px; font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; margin: 8px 0 12px;}
.risk-bar {height: 10px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden;}
.risk-bar span {display: block; height: 100%; border-radius: 999px;}
.band-low .risk-value {color: #059669;} .band-low .risk-bar span {background: #10b981;}
.band-low .risk-word {background: rgba(16,185,129,.15); color: #059669;}
.band-mid .risk-value {color: #d97706;} .band-mid .risk-bar span {background: #f59e0b;}
.band-mid .risk-word {background: rgba(245,158,11,.15); color: #d97706;}
.band-high .risk-value {color: #dc2626;} .band-high .risk-bar span {background: #ef4444;}
.band-high .risk-word {background: rgba(239,68,68,.15); color: #dc2626;}
.pills {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.pill {font-size: 13px; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--border-color-primary);}
.conf-high {background: rgba(20,184,166,.15); border-color: rgba(20,184,166,.4);}
.conf-medium {background: rgba(148,163,184,.15);}
.conf-low {background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.4);}
.sub {font-size: 13px; color: var(--body-text-color-subdued); margin: 2px 0 10px;}
.tiles {display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;}
.tile {padding: 14px; border-radius: 12px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.tile-v {font-size: 22px; font-weight: 650;} .tile-k {font-size: 12px; color: var(--body-text-color-subdued);}
.panel {padding: 18px 20px; border-radius: 14px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); height: 100%;}
.panel-h {font-weight: 650; margin-bottom: 8px;}
.list {margin: 0; padding-left: 0; list-style: none;} .list li {padding: 6px 0; border-bottom: 1px dashed var(--border-color-primary);}
.list li:last-child {border-bottom: 0;} .actions {list-style: decimal; padding-left: 20px;}
.arrow {display: inline-block; width: 20px; font-size: 11px;} .arrow.up {color: #dc2626;} .arrow.down {color: #059669;}
.muted {color: var(--body-text-color-subdued);}
.foot {font-size: 12px; color: var(--body-text-color-subdued); text-align: center; padding: 8px 0 4px;}
"""

THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])

with gr.Blocks(title="Port congestion risk") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=True):
        port = gr.Dropdown([(f"{p}  ·  {(TEST.port == p).sum()} arrivals", p) for p in PORTS], value=PORTS[0],
                           label="Port", info="US port the ship is heading to", scale=2)
        show = gr.Radio(SHOW, value=SHOW[0], label="Show", info="Filter by what actually happened", scale=3)
    ship = gr.Dropdown(label="Arriving ship", info="Held-out arrivals, newest first. Type to search by date, size or MMSI.",
                       filterable=True)
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            card = gr.HTML()
        with gr.Column(scale=4):
            context = gr.HTML()
    with gr.Row(equal_height=True):
        why = gr.HTML()
        todo = gr.HTML()
    with gr.Accordion("What if conditions were different?", open=False):
        with gr.Row():
            waiting = gr.Slider(0, int(DF.waiting_now.max()), step=1, label="Ships waiting")
            gust = gr.Slider(0, 70, step=1, label="Max gust over the past 24 h (kn)")
            recent = gr.Slider(0, int(DF.recent_wait_h.max()), step=1, label="Median wait this week (h)")
        reset = gr.Button("Reset to actual conditions", variant="secondary", size="sm")
    queue = gr.LinePlot(x="Arrival", y="Ships waiting", title=f"Ships waiting at the port ({LEAD_H} h before each arrival), Aug – Oct 2021",
                        height=260)
    gr.HTML(f'<div class="foot">Historical replay on NOAA AIS data; scoring new ships live needs a live AIS feed. '
            f'Predictions use only what was known {LEAD_H} h before arrival.</div>')

    outputs = [card, context, why, todo]
    port.change(pick_port, [port, show], [ship, queue])
    show.change(pick_port, [port, show], [ship, queue])
    ship.change(pick_ship, ship, outputs + [waiting, gust, recent])
    for s in (waiting, gust, recent):
        s.release(render, [ship, waiting, gust, recent], outputs)
    reset.click(pick_ship, ship, outputs + [waiting, gust, recent])
    demo.load(pick_port, [port, show], [ship, queue])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
