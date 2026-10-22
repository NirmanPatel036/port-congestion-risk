# Port congestion risk from AIS data

Predicts the chance that a ship arriving at a US port waits more than 24 h for a berth, gives a confidence level, explains the main drivers and suggests what a planner can do about them.

Data: NOAA Marine Cadastre AIS for Aug-Oct 2019 (a normal year) and Aug 2020-Oct 2021 (the US port backlog building up and peaking) at Los Angeles/Long Beach, Oakland, Savannah, Houston and New York/New Jersey, plus Open-Meteo hourly weather.

## How it works

1. **`src/ais.py`** downloads each day of AIS and keeps cargo ships and tankers over 100 m within each port's zone.
2. **`src/calls.py`** turns pings into port calls: when the ship arrived, when it reached a berth, and how long it waited.
   A ship counts as berthed when it sits still (90% of pings within 50 m) for 2+ h; anchored ships swing 100-350 m. Movement is trusted over the crew-entered AIS status, which is often wrong.
3. **`src/weather.py`** fetches hourly wind, gusts and rain for each port.
4. **`src/features.py`** describes what a planner could see **24 h before the ship arrives**, while there is still time to slow-steam or re-plan: ships waiting, berth throughput in the last 7 days, backlog in days (queue / berthing rate), recent waits, weather observed over the past 24 h, the ship's ETA (local time, weekday) and vessel size. Only events before that cutoff are counted. `--lead-h 0` builds the same features at the moment of arrival, kept as a baseline.
5. **`src/train.py`** trains on arrivals before 27 Sep 2021, calibrates on 27 Sep-9 Oct and tests on 10-31 Oct 2021 (fixed dates in `src/config.py`). It compares a one-line rule, logistic regression and a calibrated ensemble of 5 XGBoost models. The spread across the 5 models is the confidence score. **`src/compare.py`** prints the 24 h-ahead model next to the arrival-time baseline. **`src/experiment.py`** checks whether more history helps: learning curves, the current Aug-Sep 2021 model against all history on the same test weeks, new vessels, and 95% intervals.
6. **`app.py`** is a Streamlit demo: pick a port and a ship to see the risk, the reasons, the suggested actions and what actually happened. You can also change conditions to see how the risk responds.

## Run

Training needs internet and about 200 GB of transient downloads for 549 days (each day is deleted after filtering, so disk use stays small). The steps below run on a Kaggle notebook with Internet enabled:

```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt
python -m src.ais --workers 4      # resumable; rerun to retry failed days
python -m src.calls
python -m src.weather
python -m src.features              # 24 h before arrival (main model)
python -m src.features --lead-h 0    # at arrival (baseline)
python -m src.train
python -m src.train --lead-h 0
python -m src.compare
python -m src.experiment             # in a notebook: %run -m src.experiment to see the plots
```

Quick trial on fewer days: `python -m src.ais --start 2021-09-01 --end 2021-09-21`.

Copy `models/`, `reports/` and `data/processed/features.parquet` back to your machine, then:

```bash
streamlit run app.py
```

**Using the app:** pick a port in the sidebar, then pick an arriving ship from the held-out weeks (10-31 Oct 2021). Its features fill in automatically from AIS and weather data as they stood 24 h before arrival. You get the risk, a confidence level, the main reasons, suggested actions and what actually happened. Use "What if conditions were different?" to change the queue, gusts or recent waits and see the risk respond. Ships outside the dataset cannot be scored; that needs a live AIS feed.

## Assumptions

- Congested means waiting more than 24 h between arriving in the port zone and reaching a berth, after subtracting the time needed to sail in at 10 kn.
- A ship still waiting when the data ends counts as congested once it has waited 24 h; otherwise it is left out.
- Arrivals in the first day of each period (wait unknown) and ships first seen already at a berth are not used as labels, but still count toward queue sizes.
- A ship is only labeled if it arrived early enough for either outcome to be seen before its period's data ends; otherwise quick berthings near the end would be kept and long waits dropped.
- A training row is dropped if its outcome was only known after validation began, and a validation row if after testing began, because that label would not exist yet in production.
- Predictions are made 24 h before arrival. The ETA is assumed known at that point, so local arrival time and weekday are features; nothing observed after the cutoff is used, including weather.
- Weather is limited to readings up to the cutoff, because archived 2021 forecasts are not available.
- The data ends in October 2021 because LA/Long Beach moved its queue 150 nm offshore in mid-November, out of range of the AIS zones used here, so later waits there would look short.
