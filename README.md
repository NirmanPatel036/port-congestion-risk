# Port congestion risk from AIS data

Predicts the chance that a ship arriving at a US port waits more than 24 h for a berth, gives a confidence level, explains the main drivers and suggests what a planner can do about them.

Data: NOAA Marine Cadastre AIS for Aug-Oct 2021 (peak of the US port backlog) at Los Angeles/Long Beach, Oakland, Savannah, Houston and New York/New Jersey, plus Open-Meteo hourly weather.

## How it works

1. **`src/ais.py`** downloads each day of AIS and keeps cargo ships and tankers over 100 m within each port's zone.
2. **`src/calls.py`** turns pings into port calls: when the ship arrived, when it reached a berth, and how long it waited.
   A ship counts as berthed when it sits still (90% of pings within 50 m) for 2+ h; anchored ships swing 100-350 m. Movement is trusted over the crew-entered AIS status, which is often wrong.
3. **`src/weather.py`** fetches hourly wind, gusts and rain for each port.
4. **`src/features.py`** describes what a planner could see at the moment of arrival: ships waiting, berth throughput in the last 7 days, backlog in days (queue / berthing rate), recent waits, weather now and in the next 24 h, local time and vessel size.
5. **`src/train.py`** trains on the earliest arrivals and tests on the latest ones. It compares a one-line rule, logistic regression and a calibrated ensemble of 5 XGBoost models. The spread across the 5 models is the confidence score.
6. **`app.py`** is a Streamlit demo: pick a port and a ship to see the risk, the reasons, the suggested actions and what actually happened. You can also change conditions to see how the risk responds.

## Run

Training needs internet and about 35 GB of transient downloads (each day is deleted after filtering). The steps below run on a Kaggle notebook with Internet enabled:

```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt
python -m src.ais --workers 4      # resumable; rerun to retry failed days
python -m src.calls
python -m src.weather
python -m src.features
python -m src.train
python -m pytest -q
```

Quick trial on fewer days: `python -m src.ais --start 2021-09-01 --end 2021-09-21`.

Copy `models/`, `reports/` and `data/processed/features.parquet` back to your machine, then:

```bash
streamlit run app.py
```

## Assumptions

- Congested means waiting more than 24 h between arriving in the port zone and reaching a berth, after subtracting the time needed to sail in at 10 kn.
- A ship still waiting when the data ends counts as congested once it has waited 24 h; otherwise it is left out.
- Arrivals in the first day of data (wait unknown) and ships first seen already at a berth are not used as labels, but still count toward queue sizes.
- Weather in the next 24 h uses observed values as a stand-in for a forecast.
- The window ends in October 2021 because LA/Long Beach moved its queue 150 nm offshore in November, out of range of the AIS data used here.
