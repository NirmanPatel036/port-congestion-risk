import pandas as pd
import requests

from .config import END, PORTS, PROCESSED, START

URL = "https://archive-api.open-meteo.com/v1/archive"


def port_weather(name, lat, lon):
    params = {"latitude": lat, "longitude": lon, "timezone": "UTC", "wind_speed_unit": "kn",
              "start_date": str((pd.Timestamp(START) - pd.Timedelta(days=8)).date()),
              "end_date": str((pd.Timestamp(END) + pd.Timedelta(days=2)).date()),
              "hourly": "wind_speed_10m,wind_gusts_10m,precipitation"}
    hourly = requests.get(URL, params=params, timeout=60).json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"wind_speed_10m": "wind_kn", "wind_gusts_10m": "gust_kn",
                                              "precipitation": "precip_mm"})
    return df.assign(port=name, time=pd.to_datetime(df.time))


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    w = pd.concat([port_weather(n, lat, lon) for n, (lat, lon, _) in PORTS.items()], ignore_index=True)
    w.to_parquet(PROCESSED / "weather.parquet", index=False)
    print(w.groupby("port")[["gust_kn", "precip_mm"]].agg(["mean", "max"]).round(1))


if __name__ == "__main__":
    main()
