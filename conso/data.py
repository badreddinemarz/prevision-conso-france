"""Download the raw data into data/raw/ and load it back as clean tables.

Sources:
- Consumption + RTE's official forecasts: eCO2mix national, consolidated/definitive data (ODRE).
- Observed weather: Open-Meteo Archive API.
- Weather as it was *forecast* 1 and 2 days ahead: Open-Meteo Previous Runs API.
  Only used at the end, to check how much our score depends on knowing the true weather.
"""

import io
import time

import pandas as pd
import requests

from conso.config import CITIES, END, FORECAST_WEATHER_START, ISSUE_HOUR, RAW_DIR, START

ECO2MIX_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "eco2mix-national-cons-def/exports/csv"
)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_VARS = ["temperature_2m", "cloud_cover", "wind_speed_10m"]
TZ = "Europe/Paris"


# ---------------------------------------------------------------- download

def _get(url: str, params: dict, retries: int = 5) -> requests.Response:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=300)
        if r.status_code == 200:
            return r
        # Open-Meteo answers 429 when the per-minute quota is exceeded
        wait = 65 if r.status_code == 429 else 10 * (attempt + 1)
        print(f"  HTTP {r.status_code}, retrying in {wait}s")
        time.sleep(wait)
    r.raise_for_status()
    return r


def download_eco2mix() -> None:
    """One CSV per year. A file that already exists is not downloaded again."""
    for year in range(pd.Timestamp(START).year, pd.Timestamp(END).year + 1):
        path = RAW_DIR / f"eco2mix_{year}.csv"
        if path.exists():
            continue
        print(f"eCO2mix {year}...")
        r = _get(ECO2MIX_URL, {
            "select": "date_heure,consommation,prevision_j1,prevision_j",
            "where": f"date_heure >= '{year}-01-01T00:00:00Z' and date_heure < '{year + 1}-01-01T00:00:00Z'",
            "order_by": "date_heure",
            "delimiter": ";",
            "timezone": "UTC",
        })
        pd.read_csv(io.StringIO(r.text), sep=";").to_csv(path, index=False)


def _download_weather(url: str, variables: list[str], start: str, end: str, prefix: str) -> None:
    for city, (lat, lon, _) in CITIES.items():
        path = RAW_DIR / f"{prefix}_{city}.csv"
        if path.exists():
            continue
        print(f"{prefix} {city}...")
        r = _get(url, {
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "hourly": ",".join(variables), "timezone": "UTC",
        })
        pd.DataFrame(r.json()["hourly"]).to_csv(path, index=False)
        time.sleep(2)


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_eco2mix()
    _download_weather(ARCHIVE_URL, WEATHER_VARS, START, END, "weather_observed")
    forecast_vars = [f"{v}_previous_day{d}" for v in WEATHER_VARS for d in (1, 2)]
    _download_weather(PREVIOUS_RUNS_URL, forecast_vars, FORECAST_WEATHER_START, END, "weather_forecast")


def download_live_snapshot(path) -> None:
    """Live (not yet revised) consumption and RTE forecasts. Only the latest weeks are available."""
    r = _get(ECO2MIX_URL.replace("eco2mix-national-cons-def", "eco2mix-national-tr"), {
        "select": "date_heure,consommation,prevision_j1,prevision_j",
        "order_by": "date_heure", "delimiter": ";", "timezone": "UTC",
    })
    pd.read_csv(io.StringIO(r.text), sep=";").dropna().to_csv(path, index=False)


# -------------------------------------------------------------------- load

def load_consumption() -> pd.DataFrame:
    """Half-hourly table indexed by UTC time: actual load and RTE's two forecasts, in MW.

    load        actual consumption
    rte_d1      RTE forecast published the day before  -> the fair benchmark for our models
    rte_d0      RTE forecast updated on the day itself -> shown for information only
    """
    df = pd.concat(pd.read_csv(p) for p in sorted(RAW_DIR.glob("eco2mix_*.csv")))
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df = df.drop_duplicates("date_heure").set_index("date_heure").sort_index()
    df.columns = ["load", "rte_d1", "rte_d0"]
    # Definitive consumption exists every 30 min (the forecasts are every 15 min)
    df = df[df.index.minute.isin([0, 30])]
    # Force a complete half-hourly grid, so a missing measurement shows up as NaN, not as a gap
    start = pd.Timestamp(START, tz=TZ)
    end = pd.Timestamp(END, tz=TZ) + pd.Timedelta("23h30min")
    return df.reindex(pd.date_range(start, end, freq="30min").tz_convert("UTC"))


def load_weather(prefix: str = "weather_observed", suffix: str = "") -> pd.DataFrame:
    """Hourly table indexed by UTC time.

    National columns (temp, cloud, wind) are population-weighted averages of the cities.
    The temperature of each city is also kept (temp_paris, temp_lyon, ...) for the
    'national average vs city by city' experiment.
    """
    total_weight = sum(w for _, _, w in CITIES.values())
    national, per_city = 0, {}
    for city, (_, _, weight) in CITIES.items():
        df = pd.read_csv(RAW_DIR / f"{prefix}_{city}.csv")
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        df = df[[v + suffix for v in WEATHER_VARS]].set_axis(["temp", "cloud", "wind"], axis=1)
        national = national + df * weight / total_weight
        per_city[f"temp_{city}"] = df["temp"]
    return national.assign(**per_city)


def load_weather_as_known() -> pd.DataFrame:
    """The weather a forecaster would have had on day D-1 at ISSUE_HOUR, for each hour of day D.

    Morning hours of day D use the forecast made 1 day before; afternoon hours use the one
    made 2 days before, because the 1-day-old forecast did not exist yet at ISSUE_HOUR.
    Before FORECAST_WEATHER_START no archived forecast exists, so observed weather is used
    (training period only, never an evaluated year).
    """
    day1 = load_weather("weather_forecast", "_previous_day1")
    day2 = load_weather("weather_forecast", "_previous_day2")
    morning = day1.index.tz_convert(TZ).hour < ISSUE_HOUR
    forecast = day2.copy()
    forecast[morning] = day1[morning]
    observed = load_weather()
    return forecast.reindex(observed.index).combine_first(observed)
