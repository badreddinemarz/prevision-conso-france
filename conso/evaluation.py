"""Scores, the correction of systematic gaps, and a test of whether a difference is real."""

import numpy as np
import pandas as pd


def scores(load: pd.Series, prediction: pd.Series) -> dict:
    error = prediction - load
    return {
        "MAPE_%": (error.abs() / load).mean() * 100,   # average error, in % of consumption
        "MAE_MW": error.abs().mean(),                  # average error, in MW
        "bias_MW": error.mean(),                       # negative = forecasts too low on average
    }


def rolling_correction(prediction, load, day, halfhour, kind: str, window: int = 28) -> pd.Series:
    """Correct a forecast with its own recent systematic gap, using only the past.

    For day D, the gap is measured on the `window` days ending at D-2 (the last full day
    known when the forecast is issued on D-1). Three shapes of correction:
      shift              add the average gap in MW
      scale              multiply by the average ratio actual / forecast
      shift_by_halfhour  add the average gap of each half-hour of the day
    """
    gap = load - prediction
    if kind == "shift":
        recent = gap.groupby(day).mean().shift(2).rolling(window, min_periods=7).mean()
        return prediction + day.map(recent).fillna(0).to_numpy()
    if kind == "scale":
        both = prediction.notna() & load.notna()
        actual = load.where(both).groupby(day).sum().shift(2).rolling(window, min_periods=7).sum()
        forecast = prediction.where(both).groupby(day).sum().shift(2).rolling(window, min_periods=7).sum()
        return prediction * day.map(actual / forecast).fillna(1).to_numpy()
    if kind == "shift_by_halfhour":
        recent = gap.groupby([day, halfhour]).mean().unstack().shift(2).rolling(window, min_periods=7).mean()
        lookup = recent.stack().reindex(pd.MultiIndex.from_arrays([day, halfhour]))
        return prediction + lookup.fillna(0).to_numpy()
    raise ValueError(kind)


def best_case_correction(prediction, load, halfhour) -> pd.Series:
    """Remove the average gap of the whole period, per half-hour. Uses the answers themselves,
    so it is not a real forecast: it is the most generous score a constant correction can give."""
    return prediction + (load - prediction).groupby(halfhour).transform("mean")


def mape_difference_interval(load, a, b, day, draws: int = 2000, seed: int = 0) -> tuple:
    """95 % interval of MAPE(a) - MAPE(b), resampling whole days (errors within a day move together)."""
    ape = pd.DataFrame({"a": (a - load).abs() / load, "b": (b - load).abs() / load, "day": day})
    daily = ape.groupby("day").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(draws, len(daily)))
    values = daily.to_numpy()[picks]                      # draws x days x (a, b, n)
    diff = (values[:, :, 0].sum(1) - values[:, :, 1].sum(1)) / values[:, :, 2].sum(1) * 100
    return np.percentile(diff, 2.5), np.percentile(diff, 97.5)
