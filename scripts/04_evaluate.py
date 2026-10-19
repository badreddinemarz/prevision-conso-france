"""Step 4: fair comparison with RTE, then error analysis. Writes its tables to results/."""

import numpy as np
import pandas as pd

from conso.config import RESULTS_DIR
from conso.data import TZ, download_live_snapshot, load_consumption
from conso.evaluation import (best_case_correction, mape_difference_interval, rolling_correction,
                              scores)

FINAL = "gbm_cities"   # chosen on the validation year in step 3
pd.set_option("display.width", 200)

pred = pd.read_csv(RESULTS_DIR / "predictions.csv", index_col=0, parse_dates=["time_utc", "day"])
pred.index = pd.to_datetime(pred.index, utc=True)

# ------------------------------------------------------------ 1. What does RTE's gap look like?
full = load_consumption()
local = full.index.tz_convert(TZ)
full["day"] = pd.DatetimeIndex(local.tz_localize(None).normalize())
full["halfhour"] = local.hour * 2 + local.minute // 30
full["gap"] = full["rte_d1"] - full["load"]

print("1. RTE day-ahead forecast minus actual consumption")
by_year = full.groupby(full["day"].dt.year).agg(gap_MW=("gap", "mean"), load_MW=("load", "mean"))
by_year["gap_%"] = by_year["gap_MW"] / by_year["load_MW"] * 100
print(by_year[["gap_MW", "gap_%"]].round(1).T.to_string())
recent = full[full["day"] >= "2024-09-01"]
season = recent.groupby(recent["day"].dt.month.map(lambda m: "winter (Nov-Mar)" if m in (11, 12, 1, 2, 3) else "rest of year"))
print("\nSince Sept 2024, by season and by time of day (MW):")
print(season["gap"].mean().round(0).to_string())
block = pd.cut(recent["halfhour"], [-1, 11, 23, 35, 47], labels=["00-06h", "06-12h", "12-18h", "18-24h"])
print(recent.groupby(block, observed=True)["gap"].mean().round(0).to_string())

# ------------------------------------------------------------ 2. Rolling correction, past only
KINDS = ["shift", "scale", "shift_by_halfhour"]
for kind in KINDS:   # RTE: corrected over its full history, then aligned on the evaluated years
    corrected = rolling_correction(full["rte_d1"], full["load"], full["day"], full["halfhour"], kind)
    pred[f"rte_d1_{kind}"] = corrected.reindex(pred.index)
    pred[f"final_{kind}"] = rolling_correction(pred[FINAL], pred["load"], pred["day"], pred["halfhour"], kind)

compared = ["load", "naive", "linear", "gbm", FINAL, "rte_d1", "rte_d0"]
pred = pred.dropna(subset=compared)   # every model is scored on exactly the same half-hours
val, test = pred[pred["fold"] == "validation"], pred[pred["fold"] == "test"].copy()

print("\n2. Which shape of correction? RTE's MAPE on the validation year:")
val_scores = {kind: scores(val["load"], val[f"rte_d1_{kind}"])["MAPE_%"] for kind in KINDS}
for kind, value in val_scores.items():
    print(f"  {kind:<20}{value:.2f}")
kind = min(val_scores, key=val_scores.get)
print(f"  -> kept: {kind} (chosen on the validation year, the one that helps RTE most)")

# ------------------------------------------------------------ 3. Results table
rows = {
    "Naive (7 days before)": "naive",
    "Linear regression": "linear",
    "Gradient boosting": "gbm",
    "Gradient boosting + cities (final)": FINAL,
    "Final, rolling correction": f"final_{kind}",
    "RTE day-ahead, raw": "rte_d1",
    "RTE day-ahead, rolling correction": f"rte_d1_{kind}",
    "RTE day-ahead, best case": "rte_best",
    "RTE same-day update (not comparable)": "rte_d0",
}
tables = {}
for name, fold in [("validation", val.copy()), ("test", test)]:
    fold["rte_best"] = best_case_correction(fold["rte_d1"], fold["load"], fold["halfhour"])
    tables[name] = pd.DataFrame({label: scores(fold["load"], fold[col]) for label, col in rows.items()}).T
    if name == "test":
        test = fold
results = pd.concat(tables, axis=1)
results.round(2).to_csv(RESULTS_DIR / "results.csv")
print(f"\n3. Results ({len(val)} half-hours in validation, {len(test)} in test)")
print(results.round(2).to_string())

print("\nIs the difference with RTE real? 95 % interval of (our MAPE - RTE's MAPE), test year:")
for label, col in [("vs RTE rolling correction", f"rte_d1_{kind}"), ("vs RTE best case", "rte_best")]:
    low, high = mape_difference_interval(test["load"], test[FINAL], test[col], test["day"])
    print(f"  {label:<28}{low:+.2f} to {high:+.2f} points")

# ------------------------------------------------------------ 4. Where are the errors? (test year)
ours, rte = FINAL, f"rte_d1_{kind}"
test["ape_ours"] = (test[ours] - test["load"]).abs() / test["load"] * 100
test["ape_rte"] = (test[rte] - test["load"]).abs() / test["load"] * 100
test["ape_naive"] = (test["naive"] - test["load"]).abs() / test["load"] * 100


def breakdown(groups, name):
    table = test.groupby(groups, observed=True).agg(
        days=("day", "nunique"), naive=("ape_naive", "mean"), ours=("ape_ours", "mean"), rte=("ape_rte", "mean"))
    table.index.name = name
    return table.round(2)


daytype = np.select(
    [test["christmas_week"] == 1, test["is_holiday"] == 1, test["is_bridge"] == 1, test["weekday"] == 5, test["weekday"] == 6],
    ["Christmas week", "Public holiday", "Bridge day", "Saturday", "Sunday"], "Working day")
daily_temp = test.groupby("day")["temp_true"].transform("mean")
temp_band = pd.cut(daily_temp, [-50, 3, 8, 15, 22, 50], labels=["< 3 °C (cold spell)", "3-8 °C", "8-15 °C", "15-22 °C", "> 22 °C (heat)"])
hour_block = pd.cut(test["halfhour"], [-1, 11, 23, 35, 47], labels=["00-06h", "06-12h", "12-18h", "18-24h"])

print("\n4. Error analysis on the test year, MAPE (%): ours vs RTE with rolling correction")
analysis = {
    "by_daytype": breakdown(daytype, "day type"),
    "by_temperature": breakdown(temp_band, "daily mean temperature"),
    "by_month": breakdown(test["day"].dt.strftime("%Y-%m"), "month"),
    "by_time_of_day": breakdown(hour_block, "time of day"),
}
for name, table in analysis.items():
    table.to_csv(RESULTS_DIR / f"errors_{name}.csv")
    print("\n" + table.to_string())

worst = test.groupby("day").agg(
    ours=("ape_ours", "mean"), rte=("ape_rte", "mean"), temp=("temp_true", "mean"),
    holiday=("is_holiday", "max"), bridge=("is_bridge", "max"), xmas=("christmas_week", "max"),
    signed_MW=(ours, lambda s: (s - test.loc[s.index, "load"]).mean()))
worst["weekday"] = worst.index.day_name()
worst = worst.sort_values("ours", ascending=False).head(10).round(1)
worst.to_csv(RESULTS_DIR / "errors_worst_days.csv")
print("\nOur 10 worst days:")
print(worst.to_string())

test.to_csv(RESULTS_DIR / "test_year_scored.csv", float_format="%.1f")

# ------------------------------------------------------------ 5. Check of the "revision" explanation
# The live dataset only keeps the latest weeks, so a snapshot is saved once and kept in results/.
snapshot = RESULTS_DIR / "live_snapshot.csv"
if not snapshot.exists():
    download_live_snapshot(snapshot)
live = pd.read_csv(snapshot, parse_dates=["date_heure"]).dropna()
live = live[live["date_heure"].dt.minute.isin([0, 30])]
live_gap = live["prevision_j1"] - live["consommation"]
print(f"\n5. RTE day-ahead forecast against LIVE consumption, {live['date_heure'].min():%d/%m/%Y} -> "
      f"{live['date_heure'].max():%d/%m/%Y} ({len(live)} half-hours)")
print(f"  average gap {live_gap.mean():+.0f} MW, MAPE {(live_gap.abs() / live['consommation']).mean() * 100:.2f} %")
print(f"  (against revised consumption on the test year: {results[('test', 'bias_MW')]['RTE day-ahead, raw']:+.0f} MW)")
