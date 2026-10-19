"""Step 2: build the dataset and check it, especially that no input looks into the future."""

import pandas as pd

from conso.config import FORECAST_WEATHER_START
from conso.data import TZ, load_weather, load_weather_as_known
from conso.features import CALENDAR, PAST_LOAD, WEATHER, build_dataset

known, observed = load_weather_as_known(), load_weather()
df = build_dataset(known)
inputs = CALENDAR + WEATHER + PAST_LOAD
print(f"Dataset: {len(df)} rows, {len(inputs)} model inputs")
print(f"Rows with at least one missing input: {df[inputs].isna().any(axis=1).mean() * 100:.2f} %")

# --- Weather: forecast vs what really happened, on the period where forecasts exist
gap = (known["temp"] - observed["temp"])[FORECAST_WEATHER_START:]
print(f"\nWeather forecast error since {FORECAST_WEATHER_START}: "
      f"{gap.abs().mean():.2f} °C on average, {(gap == 0).mean() * 100:.1f} % of hours fell back on observed weather")


def at(column: str, clock_time: str) -> float:
    return df[column].loc[pd.Timestamp(clock_time, tz=TZ).tz_convert("UTC")]


print("\nNo-leak check (value given to the model  vs  value looked up by hand):")
for target, source, why in [
    ("2025-01-15 09:00", "2025-01-14 09:00", "morning   -> D-1, known at noon"),
    ("2025-01-15 18:00", "2025-01-13 18:00", "afternoon -> D-2, D-1 18:00 not known yet"),
    ("2025-03-31 18:00", "2025-03-29 18:00", "after spring clock change"),
    ("2025-10-27 09:00", "2025-10-26 09:00", "after autumn clock change"),
]:
    print(f"  {target}: load_recent = {at('load_recent', target):.0f} | load on {source} = {at('load', source):.0f}  ({why})")
print(f"  2025-01-15: load_d1_last_known = {at('load_d1_last_known', '2025-01-15 18:00'):.0f} | "
      f"load on 2025-01-14 11:30 = {at('load', '2025-01-14 11:30'):.0f}")

print("\nCorrelation with consumption:")
print(df[["load_recent", "load_d7", "load_d2", "heating_smooth", "temp"]].corrwith(df["load"]).round(2).to_string())
