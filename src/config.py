from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

AIS_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip"

# name: (lat, lon, radius_km) around the terminals; radius covers the anchorages
PORTS = {
    "Los Angeles/Long Beach": (33.74, -118.24, 60),
    "Oakland": (37.80, -122.32, 40),
    "Savannah": (32.08, -81.05, 60),
    "Houston": (29.55, -94.95, 70),
    "New York/New Jersey": (40.62, -74.08, 40),
}

TZ = {
    "Los Angeles/Long Beach": "America/Los_Angeles",
    "Oakland": "America/Los_Angeles",
    "Savannah": "America/New_York",
    "Houston": "America/Chicago",
    "New York/New Jersey": "America/New_York",
}

# AIS periods, each processed on its own so no visit spans a gap. Nothing after Oct 2021: in mid-Nov 2021
# LA/LB moved its queue 150 nm offshore, out of these zones, so waits there would look short.
PERIODS = [("2019-08-01", "2019-10-31"),   # a normal year, before the backlog
           ("2020-08-01", "2021-10-31")]   # backlog builds up and peaks
START, END = PERIODS[0][0], PERIODS[-1][1]
# fixed split dates so every model is validated and tested on the same weeks
VAL_START, TEST_START = "2021-09-27", "2021-10-10"

MIN_LENGTH_M = 100     # ocean-going cargo ships and tankers, not tugs or barges
STOP_KNOTS = 0.5
MIN_BERTH_H = 2
BERTH_SPREAD_M = 50    # moored ships stay within ~10 m; anchored ones swing 100-350 m
JUMP_KM = 1            # a bigger move between pings means the ship relocated while AIS was quiet
NEW_VISIT_GAP_H = 24
APPROACH_KNOTS = 10    # time to sail from zone edge to berth is not counted as waiting
DELAY_H = 24           # congested = waited more than a day for a berth
LEAD_H = 24            # predict this many hours before the ship arrives; 0 gives the arrival-time baseline


def tag(lead_h):
    """File-name suffix: the main model has none, other lead times are kept side by side."""
    return "" if lead_h == LEAD_H else f"_lead{lead_h}h"
