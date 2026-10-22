import numpy as np
import pandas as pd

from .config import (APPROACH_KNOTS, BERTH_SPREAD_M, DELAY_H, END, JUMP_KM, MIN_BERTH_H, NEW_VISIT_GAP_H, PORTS,
                     PROCESSED, RAW, START, STOP_KNOTS)

KM_PER_NM = 1.852
H = pd.Timedelta(hours=1)


def km_between(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load_pings():
    df = pd.concat((pd.read_parquet(p) for p in sorted(RAW.glob("*.parquet"))), ignore_index=True)
    return df.rename(columns={"MMSI": "mmsi", "BaseDateTime": "t", "LAT": "lat", "LON": "lon", "SOG": "sog",
                              "Status": "status", "VesselType": "vtype", "Length": "length", "Draft": "draft"})


def clean(df):
    df = df.drop_duplicates(["mmsi", "t"]).sort_values(["mmsi", "t"], ignore_index=True)
    hours = df.t.diff().dt.total_seconds() / 3600
    knots = km_between(df.lat.shift(), df.lon.shift(), df.lat, df.lon) / KM_PER_NM / hours.clip(lower=1 / 60)
    return df[~(df.mmsi.eq(df.mmsi.shift()) & (knots > 40))].reset_index(drop=True)  # GPS jumps


def split(df, gap_h, extra=None):
    gap = df.t.diff().dt.total_seconds() / 3600 > gap_h
    new = ~df.mmsi.eq(df.mmsi.shift()) | ~df.port.eq(df.port.shift()) | gap
    return (new if extra is None else new | extra).cumsum()


def stop_segments(df):
    stopped = df.sog < STOP_KNOTS
    # a ship can move from anchorage to berth with no moving pings in between, so a jump also ends a stop
    jumped = km_between(df.lat.shift(), df.lon.shift(), df.lat, df.lon) > JUMP_KM
    df["seg"] = split(df, 3, stopped.ne(stopped.shift()) | jumped)
    s = df[stopped].copy()
    g = s.groupby("seg")
    s["d_m"] = 1000 * km_between(s.lat, s.lon, g.lat.transform("median"), g.lon.transform("median"))
    s["anchor"] = s.status.eq(1)
    segs = s.groupby("seg").agg(part=("part", "first"), start=("t", "min"), end=("t", "max"),
                                lat=("lat", "median"), lon=("lon", "median"), anchor_share=("anchor", "mean"))
    segs["spread_m"] = s.groupby("seg").d_m.quantile(0.9)
    segs["hours"] = (segs.end - segs.start).dt.total_seconds() / 3600
    # a moored ship sits still; an anchored or drifting one swings, whatever status the crew typed in
    segs["berth"] = (segs.hours >= MIN_BERTH_H) & (segs.spread_m < BERTH_SPREAD_M) & (segs.anchor_share < 0.5)
    return segs


def presence_parts(df, segs):
    p = df.groupby("part").agg(mmsi=("mmsi", "first"), port=("port", "first"), first=("t", "min"), last=("t", "max"),
                               lat0=("lat", "first"), lon0=("lon", "first"), length=("length", "median"),
                               draft=("draft", "median"), vtype=("vtype", "first"), seg0=("seg", "first"))
    p["stop_h"] = segs.groupby("part").hours.sum().reindex(p.index, fill_value=0)
    berths = segs[segs.berth].sort_values("start").groupby("part")[["start", "end", "lat", "lon"]].first()
    p = p.join(berths.add_prefix("berth_"))
    p["starts_berthed"] = p.seg0.map(segs.berth).eq(True)  # AIS switched on at the dock: true arrival unknown
    return p.drop(columns="seg0")


def merge_into_visits(p):
    # ships waiting far offshore drop out of AIS range; a gap is not a new visit until the ship has berthed
    p = p.sort_values(["mmsi", "first"])
    ids, v, key, last, berthed, waiting = [], -1, None, None, True, False
    for r in p.itertuples():
        same_ship = (r.mmsi, r.port) == key
        if not (same_ship and not berthed and waiting and r.first - last <= pd.Timedelta(days=7)):
            v, berthed, waiting = v + 1, False, False
        ids.append(v)
        key, last = (r.mmsi, r.port), r.last
        berthed |= pd.notna(r.berth_start)
        waiting |= r.stop_h >= MIN_BERTH_H or (r.last - r.first) >= 12 * H
    p["visit"] = ids
    return p.groupby("visit").agg(
        mmsi=("mmsi", "first"), port=("port", "first"), arrival=("first", "min"), last=("last", "max"),
        lat0=("lat0", "first"), lon0=("lon0", "first"), length=("length", "median"), draft=("draft", "median"),
        vtype=("vtype", "first"), stop_h=("stop_h", "sum"), starts_berthed=("starts_berthed", "first"),
        berth_start=("berth_start", "first"), berth_end=("berth_end", "first"),
        berth_lat=("berth_lat", "first"), berth_lon=("berth_lon", "first"))


def label(v, start, end):
    plat, plon = v.port.map(lambda p: PORTS[p][0]), v.port.map(lambda p: PORTS[p][1])
    sail_h = km_between(v.lat0, v.lon0, v.berth_lat.fillna(plat), v.berth_lon.fillna(plon)) / (APPROACH_KNOTS * KM_PER_NM)
    v["wait_h"] = ((v.berth_start.fillna(end) - v.arrival).dt.total_seconds() / 3600 - sail_h).clip(lower=0)

    berthed = v.berth_start.notna()
    v["censored"] = ~berthed & (v["last"] >= end - 6 * H)  # still waiting when the data ends
    v = v[berthed | (v.stop_h >= MIN_BERTH_H) | (v.censored & (v.wait_h > DELAY_H))].copy()
    # a ship still waiting past the threshold is congested even though we never see it berth
    v["congested"] = np.where(v.berth_start.notna() | (v.censored & (v.wait_h > DELAY_H)),
                              (v.wait_h > DELAY_H).astype(float), np.nan)
    v.loc[v.berth_start.isna() & ~v.censored, "wait_h"] = np.nan
    v["labeled"] = v.congested.notna() & ~v.starts_berthed & (v.arrival >= start + NEW_VISIT_GAP_H * H)
    v["queue_exit"] = v.berth_start.fillna(v["last"])
    return v.reset_index(drop=True)


def build_calls(pings, start, end):
    df = clean(pings)
    df["part"] = split(df, NEW_VISIT_GAP_H)
    segs = stop_segments(df)
    return label(merge_into_visits(presence_parts(df, segs)), start, end)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    calls = build_calls(load_pings(), pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=1))
    calls.to_parquet(PROCESSED / "calls.parquet", index=False)
    summary = calls[calls.labeled].groupby("port").agg(calls=("mmsi", "size"), congested_rate=("congested", "mean"),
                                                       median_wait_h=("wait_h", "median"))
    print(f"{len(calls)} visits, {calls.labeled.sum()} labeled\n{summary.round(2)}")


if __name__ == "__main__":
    main()
