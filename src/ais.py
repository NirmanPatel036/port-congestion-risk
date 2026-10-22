import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import requests

from .config import AIS_URL, MIN_LENGTH_M, PERIODS, PORTS, RAW

COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Status", "VesselType", "Length", "Draft"]


def port_of(lat, lon):
    port = np.full(len(lat), None, dtype=object)
    for name, (plat, plon, km) in PORTS.items():
        dy = (lat - plat) * 111.0
        dx = (lon - plon) * 111.0 * np.cos(np.radians(plat))
        port[(port == None) & (np.hypot(dx, dy) <= km)] = name  # noqa: E711
    return port


def fetch_day(day):
    try:
        return _fetch_day(day)
    except Exception as e:  # one bad day should not kill a 90-day run; rerun picks it up
        return f"{day:%Y-%m-%d}", f"FAILED: {e}"


def _fetch_day(day):
    out = RAW / f"{day:%Y-%m-%d}.parquet"
    if out.exists():
        return out.name, "cached"
    zip_path = RAW / f"{day:%Y-%m-%d}.zip"
    with requests.get(AIS_URL.format(y=day.year, m=day.month, d=day.day), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)

    keep = []
    for chunk in pd.read_csv(zip_path, usecols=COLS, chunksize=1_000_000, on_bad_lines="skip"):
        chunk = chunk[chunk.VesselType.between(70, 89) & (chunk.Length >= MIN_LENGTH_M)]
        chunk = chunk.assign(port=port_of(chunk.LAT.values, chunk.LON.values)).dropna(subset=["port"])
        keep.append(chunk)
    df = pd.concat(keep, ignore_index=True)
    df["BaseDateTime"] = pd.to_datetime(df.BaseDateTime)
    df.to_parquet(out, index=False)
    zip_path.unlink()
    return out.name, len(df)


def main():
    ap = argparse.ArgumentParser(description="Download NOAA AIS days and keep ships near the study ports")
    ap.add_argument("--start", help="default: every day in config.PERIODS")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.start:
        days = pd.date_range(args.start, args.end or args.start, freq="D")
    else:
        days = pd.DatetimeIndex([d for s, e in PERIODS for d in pd.date_range(s, e, freq="D")])
    with ProcessPoolExecutor(args.workers) as pool:
        for name, n in pool.map(fetch_day, days):
            print(name, n, flush=True)


if __name__ == "__main__":
    main()
