import argparse

import numpy as np
import pandas as pd

from .config import LEAD_H, PROCESSED, START, TZ, tag

FEATURES = ["port", "waiting_now", "backlog_days", "arrivals_24h", "berths_7d", "at_berth_now", "recent_wait_h",
            "gust_kn", "gust_max_prev_24h_kn", "precip_prev_24h_mm", "hour_local", "is_weekend", "length_m",
            "draft_m", "is_tanker"]
RISING = ["waiting_now", "backlog_days", "arrivals_24h", "recent_wait_h", "gust_max_prev_24h_kn"]  # only push risk up
DAY = np.timedelta64(1, "D")


def backlog_days(waiting, berths_7d):
    # Little's law: queue length / service rate = expected wait
    return waiting / np.maximum(berths_7d / 7, 0.5)


def port_state(g):
    """What a port planner can see at each call's cutoff time, counting only events before it."""
    t = g.cutoff.values
    times = lambda col: np.sort(g[col].dropna().values)
    before = lambda s, x: np.searchsorted(s, x, side="left")
    arr, q_exit, b_in, b_out = times("arrival"), times("queue_exit"), times("berth_start"), times("berth_end")

    out = pd.DataFrame(index=g.index)
    out["waiting_now"] = before(arr, t) - before(q_exit, t)
    out["at_berth_now"] = before(b_in, t) - before(b_out, t)
    out["arrivals_24h"] = before(arr, t) - before(arr, t - DAY)
    out["berths_7d"] = before(b_in, t) - before(b_in, t - 7 * DAY)

    done = g[g.berth_start.notna() & ~g.starts_berthed]
    bs, wh = done.berth_start.values, done.wait_h.values
    recent = []
    for x in t:
        last_week = wh[(bs < x) & (bs >= x - 7 * DAY)]
        recent.append(np.median(last_week) if len(last_week) else np.nan)
    out["recent_wait_h"] = recent
    return out


def add_weather(df, w):
    # readings up to the cutoff only: each hourly value covers the hour before its timestamp
    w = w.sort_values(["port", "time"]).copy()
    past = lambda s, how: getattr(s.rolling(24, min_periods=1), how)()
    w["gust_max_prev_24h_kn"] = w.groupby("port").gust_kn.transform(lambda s: past(s, "max"))
    w["precip_prev_24h_mm"] = w.groupby("port").precip_mm.transform(lambda s: past(s, "sum"))
    df["hour"] = df.cutoff.dt.floor("h")
    cols = ["port", "time", "gust_kn", "gust_max_prev_24h_kn", "precip_prev_24h_mm"]
    return df.merge(w[cols], left_on=["port", "hour"], right_on=["port", "time"], how="left").drop(columns=["hour", "time"])


def build(calls, weather, lead_h=LEAD_H):
    calls = calls.sort_values("arrival", ignore_index=True)
    calls["cutoff"] = calls.arrival - pd.Timedelta(hours=lead_h)  # the moment the prediction is made
    if "period" not in calls:
        calls["period"] = START
    df = calls.join(pd.concat([port_state(g) for _, g in calls.groupby(["period", "port"])]))
    df["backlog_days"] = backlog_days(df.waiting_now, df.berths_7d)
    df = add_weather(df, weather)

    for port, idx in df.groupby("port").groups.items():
        local = df.loc[idx, "arrival"].dt.tz_localize("UTC").dt.tz_convert(TZ[port])
        df.loc[idx, "hour_local"] = local.dt.hour
        df.loc[idx, "is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    # time of day and ship details come from the ETA and AIS voyage data, both known before arrival
    df["length_m"], df["draft_m"] = df.length, df.draft
    df["is_tanker"] = df.vtype.between(80, 89).astype(int)

    # a week of history before the earliest cutoff in each period; fixed so every lead time scores the same ships
    warm = pd.to_datetime(df.period) + pd.Timedelta(days=7, hours=max(lead_h, LEAD_H))
    keep = df.labeled & (df.arrival >= warm)
    cols = ["mmsi", "period", "arrival", "label_time", "wait_h", "censored", "congested"]
    return df.loc[keep, cols + FEATURES].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Build features as seen LEAD hours before each arrival")
    ap.add_argument("--lead-h", type=int, default=LEAD_H, help="0 = at arrival (baseline)")
    lead_h = ap.parse_args().lead_h
    df = build(pd.read_parquet(PROCESSED / "calls.parquet"), pd.read_parquet(PROCESSED / "weather.parquet"), lead_h)
    out = PROCESSED / f"features{tag(lead_h)}.parquet"
    df.to_parquet(out, index=False)
    print(out.name, df.shape, "\n", df.groupby("port").congested.agg(["size", "mean"]).round(2))


if __name__ == "__main__":
    main()
