import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import requests

from .config import AIS_URL, MIN_LENGTH_M, PERIODS, PORTS, RAW

COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Status", "VesselType", "Length", "Draft"]
NUMERIC = [c for c in COLS if c != "BaseDateTime"]


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


def download(url, path, tries=4):
    for attempt in range(tries):  # NOAA sometimes answers 504 under load; wait and try again
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            return
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(30 * (attempt + 1))


def filter_csv(path, chunksize=500_000):
    keep = []
    for chunk in pd.read_csv(path, usecols=COLS, chunksize=chunksize, on_bad_lines="skip", dtype=str):
        # some 2020 files have stray text rows in numeric columns; read as text, convert, drop what fails
        chunk[NUMERIC] = chunk[NUMERIC].apply(pd.to_numeric, errors="coerce")
        chunk = chunk.dropna(subset=["MMSI", "LAT", "LON", "BaseDateTime"])
        chunk = chunk[chunk.VesselType.between(70, 89) & (chunk.Length >= MIN_LENGTH_M)]
        chunk = chunk.assign(port=port_of(chunk.LAT.values, chunk.LON.values)).dropna(subset=["port"])
        keep.append(chunk)
    df = pd.concat(keep, ignore_index=True)
    df["MMSI"] = df.MMSI.astype("int64")
    df["BaseDateTime"] = pd.to_datetime(df.BaseDateTime, errors="coerce")
    return df.dropna(subset=["BaseDateTime"])


def _fetch_day(day):
    out = RAW / f"{day:%Y-%m-%d}.parquet"
    if out.exists():
        return out.name, "cached"
    zip_path = RAW / f"{day:%Y-%m-%d}.zip"
    download(AIS_URL.format(y=day.year, m=day.month, d=day.day), zip_path)
    df = filter_csv(zip_path)
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
