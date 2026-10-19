"""Turn the raw tables into one half-hourly dataset: target, RTE forecasts and model inputs.

The no-leak rule (same conditions as RTE's day-ahead forecast)
---------------------------------------------------------------
The forecast for day D is issued on day D-1 at ISSUE_HOUR (French time).
Every input built from past consumption stops at that moment.
The calendar of day D is known in advance; its weather is the weather *forecast* available
at that moment (see data.load_weather_as_known).
"""

import holidays
import numpy as np
import pandas as pd

from conso.config import CITIES, ISSUE_HOUR
from conso.data import TZ, load_consumption

CITY_TEMPS = [f"temp_{city}" for city in CITIES]


def same_time_days_before(series: pd.Series, local_index: pd.DatetimeIndex, days: int) -> np.ndarray:
    """Value of `series` at the same clock time `days` days earlier.

    Works on French clock time, not on elapsed hours, so 18:00 always looks up 18:00,
    even across the two clock changes of the year.
    """
    clock = local_index.tz_localize(None) - pd.Timedelta(days=days)
    earlier = clock.tz_localize(TZ, ambiguous=np.ones(len(clock), dtype=bool), nonexistent="shift_forward")
    return series.reindex(earlier.tz_convert("UTC")).to_numpy()


def build_dataset(weather: pd.DataFrame) -> pd.DataFrame:
    df = load_consumption()
    local = df.index.tz_convert(TZ)
    day = pd.DatetimeIndex(local.tz_localize(None).normalize())
    df["day"] = day

    # ---- Weather: hourly -> half-hourly by interpolation
    w = weather.reindex(df.index.union(weather.index)).interpolate("time", limit=4).reindex(df.index)
    df[w.columns] = w
    # Buildings take time to cool down or warm up: consumption follows a *smoothed* temperature
    df["temp_smooth_1d"] = df["temp"].ewm(halflife=48).mean()    # ~1 day of memory
    df["temp_smooth_3d"] = df["temp"].ewm(halflife=144).mean()   # ~3 days of memory
    df["heating"] = (15 - df["temp"]).clip(lower=0)              # degrees below 15 °C
    df["heating_smooth"] = (15 - df["temp_smooth_1d"]).clip(lower=0)
    df["cooling"] = (df["temp"] - 22).clip(lower=0)              # degrees above 22 °C

    # ---- Calendar, in French local time
    df["halfhour"] = local.hour * 2 + local.minute // 30         # 0..47
    df["weekday"] = local.dayofweek                              # 0 = Monday
    df["dayofyear"] = local.dayofyear
    df["month"] = local.month
    french_holidays = holidays.France(years=range(day.year.min() - 1, day.year.max() + 2))
    holiday_dates = pd.DatetimeIndex(sorted(french_holidays.keys()))
    df["is_holiday"] = day.isin(holiday_dates).astype(int)
    df["holiday_tomorrow"] = (day + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    df["holiday_yesterday"] = (day - pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    # "Pont": Friday after a Thursday holiday, or Monday before a Tuesday holiday
    df["is_bridge"] = (
        ((df["weekday"] == 4) & (df["holiday_yesterday"] == 1))
        | ((df["weekday"] == 0) & (df["holiday_tomorrow"] == 1))
    ).astype(int)
    df["christmas_week"] = (
        ((df["month"] == 12) & (local.day >= 24)) | ((df["month"] == 1) & (local.day == 1))
    ).astype(int)
    # 0 = working day, 1 = Saturday, 2 = Sunday or public holiday
    df["daytype"] = np.select(
        [(df["weekday"] == 6) | (df["is_holiday"] == 1), df["weekday"] == 5], [2, 1], 0
    )

    # ---- Past consumption up to day D-2 (whole days, fully known)
    load = df["load"]
    for d in (2, 7, 14):
        df[f"load_d{d}"] = same_time_days_before(load, local, d)
    daily = df.groupby("day")["load"].agg(["mean", "min", "max"])
    for stat in ("mean", "min", "max"):
        df[f"load_d2_day{stat}"] = (day - pd.Timedelta(days=2)).map(daily[stat]).to_numpy()
    # Tell the model when its reference days were unusual
    df["d7_was_holiday"] = (day - pd.Timedelta(days=7)).isin(holiday_dates).astype(int)
    df["d2_daytype"] = same_time_days_before(df["daytype"], local, 2)

    # ---- Consumption of the morning of day D-1, known when the forecast is issued at ISSUE_HOUR
    cutoff = ISSUE_HOUR * 2                                   # first half-hour NOT yet known
    yesterday = day - pd.Timedelta(days=1)
    known_d1 = same_time_days_before(load, local, 1)
    # Most recent known value at the same clock time: D-1 for the morning, D-2 for the afternoon
    df["load_recent"] = np.where(df["halfhour"] < cutoff, known_d1, df["load_d2"])
    morning = df[df["halfhour"] < cutoff]
    df["load_d1_morning_mean"] = yesterday.map(morning.groupby("day")["load"].mean()).to_numpy()
    last = df[df["halfhour"] == cutoff - 1].set_index("day")["load"]
    df["load_d1_last_known"] = yesterday.map(last[~last.index.duplicated()]).to_numpy()
    return df


# Groups of inputs, so each model (and each experiment) says exactly what it uses
CALENDAR = ["halfhour", "weekday", "dayofyear", "month", "is_holiday", "holiday_tomorrow",
            "holiday_yesterday", "is_bridge", "christmas_week", "daytype"]
WEATHER = ["temp", "cloud", "wind", "temp_smooth_1d", "temp_smooth_3d", "heating",
           "heating_smooth", "cooling"]
PAST_LOAD_UP_TO_D2 = ["load_d2", "load_d7", "load_d14", "load_d2_daymean", "load_d2_daymin",
                      "load_d2_daymax", "d7_was_holiday", "d2_daytype"]
PAST_LOAD = PAST_LOAD_UP_TO_D2 + ["load_recent", "load_d1_morning_mean", "load_d1_last_known"]
