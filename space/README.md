---
title: Port Congestion Risk
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Will a ship wait over 24 h for a berth? Forecast 24 h ahead
---

# Port congestion risk

Pick a US port and an arriving ship to see the chance it waits more than 24 hours for a berth, how confident the
model is, the main reasons, and what a planner can still do. Features use only what was known 24 hours before
arrival. Built on NOAA AIS data (Aug 2019 – Oct 2021) for Los Angeles/Long Beach, Oakland, Savannah, Houston and
New York/New Jersey, with Open-Meteo weather and a calibrated ensemble of 5 XGBoost models.

This is a static page: every test ship is scored in advance with the trained model, and the what-if sliders re-run the same model (exported as plain decision trees) in your browser. Code: https://github.com/NirmanPatel036/port-congestion-risk
