"""Step 1: download everything, then print a short health check of what we got."""

from conso.data import download_all, load_consumption, load_weather

download_all()

conso = load_consumption()
weather = load_weather()

print(f"\nConsumption: {len(conso)} half-hours, {conso.index[0]} -> {conso.index[-1]}")
print("Missing values (%):")
print((conso.isna().mean() * 100).round(2).to_string())
print("\nYearly average (MW):")
print(conso.groupby(conso.index.year).mean().round(0).to_string())

print(f"\nWeather: {len(weather)} hours, missing national temperature: {weather['temp'].isna().sum()}")
print(weather.describe().loc[["min", "mean", "max"], ["temp", "cloud", "wind", "temp_lille", "temp_nice"]].round(1).to_string())

forecast = load_weather("weather_forecast", "_previous_day1")
both = weather[["temp"]].join(forecast[["temp"]], rsuffix="_forecast", how="inner").dropna()
print(f"\nForecast weather: {len(forecast)} hours, {forecast.index[0]} -> {forecast.index[-1]}")
print(f"Mean absolute gap between day-ahead forecast and observed temperature: "
      f"{(both['temp'] - both['temp_forecast']).abs().mean():.2f} °C")
