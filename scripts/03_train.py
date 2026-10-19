"""Step 3: walk-forward training. For each evaluated year, train on everything before it,
predict the year, and save all predictions to results/predictions.csv.
"""

import pandas as pd

from conso import models
from conso.config import FOLDS, FORECAST_WEATHER_START, RESULTS_DIR
from conso.data import load_weather, load_weather_as_known
from conso.features import CALENDAR, CITY_TEMPS, PAST_LOAD_UP_TO_D2, WEATHER, build_dataset

df = build_dataset(load_weather_as_known())
df_true_weather = build_dataset(load_weather())   # side experiment only

KEEP = ["load", "rte_d1", "rte_d0", "day", "temp", "halfhour", "weekday", "is_holiday",
        "is_bridge", "christmas_week"]
folds = []
for name, (start, end) in zip(["validation", "test"], FOLDS):
    is_train = df["day"] < start
    is_test = (df["day"] >= start) & (df["day"] <= end)
    train, test = df[is_train], df[is_test]
    out = test[KEEP].copy()
    out.insert(0, "fold", name)

    # The three models of the brief
    out["naive"] = models.naive(test)
    out["linear"] = models.linear(train, test)
    out["gbm"] = models.gradient_boosting(train, test)

    # Variants to choose from (decided on the validation year only)
    out["gbm_cities"] = models.gradient_boosting(train, test, models.GBM_INPUTS + CITY_TEMPS)
    recent = train[train["day"] >= FORECAST_WEATHER_START]
    out["gbm_forecast_weather_only"] = models.gradient_boosting(recent, test)

    # Side experiments: what is each piece of information worth?
    out["gbm_info_d2"] = models.gradient_boosting(train, test, CALENDAR + WEATHER + PAST_LOAD_UP_TO_D2)
    out["gbm_true_weather"] = models.gradient_boosting(df_true_weather[is_train], df_true_weather[is_test])
    out["temp_true"] = df_true_weather.loc[is_test, "temp"]

    folds.append(out)
    print(f"{name}: trained on {is_train.sum()} half-hours (until {start}), predicted {is_test.sum()}")

RESULTS_DIR.mkdir(exist_ok=True)
pd.concat(folds).rename_axis("time_utc").to_csv(RESULTS_DIR / "predictions.csv", float_format="%.1f")

# First look, validation year only. The test year stays closed until the evaluation step.
val = folds[0].dropna()
mape = lambda col: (val[col] - val["load"]).abs().div(val["load"]).mean() * 100
print(f"\nValidation year, MAPE (%) on {len(val)} half-hours:")
for col in ["naive", "linear", "gbm", "gbm_cities", "gbm_forecast_weather_only",
            "gbm_info_d2", "gbm_true_weather", "rte_d1"]:
    print(f"  {col:<28}{mape(col):.2f}")
