import math

JIT = "Slow-steam so the ship arrives as a berth frees up instead of joining the queue (saves fuel and anchorage time)."
WEATHER = "Expect pilot and crane weather holds: add a buffer and avoid tight onward connections."
BIG_SHIP = "Large ships fit only a few deep-water berths and cranes: book the berth window early."


def num(x, fmt=".0f"):
    return "unknown" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


# feature -> (what the planner sees, what they can do about it)
TEXT = {
    "waiting_now": (lambda r: f"{num(r.waiting_now)} ships already waiting for a berth", JIT),
    "backlog_days": (lambda r: f"Queue equals about {num(r.backlog_days, '.1f')} days of berthing at this week's pace",
                     "Ask the terminal for a fixed berth window; with over 2 days of backlog, slow-steam to arrive as the queue clears."),
    "arrivals_24h": (lambda r: f"{num(r.arrivals_24h)} large ships arrived in the 24 h before the forecast",
                     "Arrivals are bunching: stagger ETAs with the terminal rather than first-come-first-served."),
    "berths_7d": (lambda r: f"{num(r.berths_7d)} ships berthed in the past 7 days",
                  "Throughput is low: check terminal labour and yard status, or consider an alternate terminal."),
    "at_berth_now": (lambda r: f"{num(r.at_berth_now)} ships occupying berths right now",
                     "Berths are full: confirm when the next one frees up before committing to an ETA."),
    "recent_wait_h": (lambda r: f"Ships berthing here this week waited a median {num(r.recent_wait_h)} h",
                      "Pad the customer ETA by at least the recent wait and prioritise time-critical cargo."),
    "gust_kn": (lambda r: f"Gusts of {num(r.gust_kn)} kn at the port when forecast", WEATHER),
    "gust_max_prev_24h_kn": (lambda r: f"Gusts up to {num(r.gust_max_prev_24h_kn)} kn at the port over the past 24 h",
                             WEATHER),
    "precip_prev_24h_mm": (lambda r: f"{num(r.precip_prev_24h_mm)} mm of rain at the port over the past 24 h",
                           "Rain slows cargo handling and leaves a backlog: plan for a longer stay at berth."),
    "hour_local": (lambda r: f"Due to arrive at {num(r.hour_local, '02.0f')}:00 local time",
                   "If speed allows, time the arrival for a weekday day shift when more labour gangs work."),
    "is_weekend": (lambda r: "Due on a weekend" if r.is_weekend else "Due on a weekday",
                   "If speed allows, time the arrival for a weekday day shift when more labour gangs work."),
    "length_m": (lambda r: f"Vessel is {num(r.length_m)} m long", BIG_SHIP),
    "draft_m": (lambda r: f"Draft of {num(r.draft_m, '.1f')} m", BIG_SHIP),
    "is_tanker": (lambda r: "Tanker" if r.is_tanker else "Cargo / container ship",
                  "Confirm the liquid-bulk berth schedule with the terminal."),
    "port": (lambda r: f"Conditions specific to {r.port}",
             "Compare the same voyage against an alternate port in the region."),
}


def explain(row, contrib, top=3, min_push=0.05):
    """Top risk-raising factors with actions, plus the factors holding risk down."""
    c = contrib.sort_values(ascending=False)
    raising = [(f, TEXT[f][0](row), TEXT[f][1]) for f in c.index[:top] if c[f] > min_push]
    lowering = [TEXT[f][0](row) for f in c.index[::-1][:2] if c[f] < -min_push]
    actions = list(dict.fromkeys(a for _, _, a in raising))
    return raising, lowering, actions
