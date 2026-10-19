"""The three models. Each function takes a training table and a table to predict, and returns predictions.

Hyperparameters are fixed in advance: nothing is tuned on the evaluated years.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from conso.features import CALENDAR, PAST_LOAD, WEATHER

LINEAR_NUMBERS = ["load_recent", "load_d7", "load_d2_daymean", "load_d1_morning_mean",
                  "load_d1_last_known", "heating", "heating_smooth", "cooling", "cloud", "wind",
                  "holiday_tomorrow", "holiday_yesterday", "is_bridge", "christmas_week",
                  "d7_was_holiday"]
LINEAR_CATEGORIES = ["halfhour_daytype", "month", "weekday"]

GBM_INPUTS = CALENDAR + WEATHER + PAST_LOAD
GBM_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1,
)


def naive(test: pd.DataFrame) -> np.ndarray:
    """Reference: the same half-hour, 7 days before."""
    return test["load_d7"].to_numpy()


def linear(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Linear regression with one daily profile per type of day (working / Saturday / Sunday-holiday)."""
    def prepare(df):
        return df.assign(halfhour_daytype=df["halfhour"] + 48 * df["daytype"])

    columns = LINEAR_NUMBERS + LINEAR_CATEGORIES
    train, test = prepare(train).dropna(subset=columns + ["load"]), prepare(test)
    model = make_pipeline(
        ColumnTransformer([
            ("numbers", StandardScaler(), LINEAR_NUMBERS),
            ("categories", OneHotEncoder(handle_unknown="ignore"), LINEAR_CATEGORIES),
        ]),
        LinearRegression(),
    ).fit(train[columns], train["load"])
    # A linear model cannot handle a missing input: leave those rows without a prediction
    complete = test[columns].notna().all(axis=1)
    prediction = np.full(len(test), np.nan)
    prediction[complete] = model.predict(test.loc[complete, columns])
    return prediction


def gradient_boosting(train: pd.DataFrame, test: pd.DataFrame, inputs: list[str] = GBM_INPUTS) -> np.ndarray:
    train = train.dropna(subset=["load"])
    model = lgb.LGBMRegressor(**GBM_PARAMS).fit(train[inputs], train["load"])
    return model.predict(test[inputs])
