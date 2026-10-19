"""Every choice that defines the experiment lives here, so nothing is hidden inside the code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

# Study period (local French dates, both ends included)
START = "2018-01-01"
END = "2026-08-31"

# Walk-forward validation: each fold is one full year the model has never seen,
# and the model is trained only on data from before that year.
# The last fold is the final test set.
FOLDS = [
    ("2024-09-01", "2025-08-31"),   # validation year: used to choose between model variants
    ("2025-09-01", "2026-08-31"),   # test year: looked at once, at the end
]

# Same conditions as RTE's day-ahead forecast: the forecast for day D is issued on day D-1
# at this hour (French time), with the consumption and the weather forecast known at that moment.
ISSUE_HOUR = 12
# Archived weather forecasts exist from this date; before it we fall back on observed weather.
FORECAST_WEATHER_START = "2024-01-01"

# Cities used to build one national weather series: (latitude, longitude, weight).
# Weights are rough metro-area populations in millions.
CITIES = {
    "paris": (48.85, 2.35, 13.1),
    "lyon": (45.76, 4.84, 2.3),
    "marseille": (43.30, 5.37, 1.9),
    "lille": (50.63, 3.06, 1.5),
    "toulouse": (43.60, 1.44, 1.5),
    "bordeaux": (44.84, -0.58, 1.4),
    "nantes": (47.22, -1.55, 1.0),
    "strasbourg": (48.58, 7.75, 0.9),
    "rennes": (48.11, -1.68, 0.8),
    "nice": (43.70, 7.27, 0.6),
}
