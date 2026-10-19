"""Step 5: the figures of the README, from the tables written by 04_evaluate.py."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from conso.config import RESULTS_DIR
from conso.data import TZ

OURS, RTE, ACTUAL = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def title(ax, text, subtitle):
    ax.set_title(text, loc="left", fontsize=13, fontweight="bold", pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10)


def save(fig, name):
    fig.savefig(RESULTS_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---- 1. Scores of every model on the test year
results = pd.read_csv(RESULTS_DIR / "results.csv", header=[0, 1], index_col=0)["test"]["MAPE_%"]
bars = [
    ("Naïf (même heure, J-7)", "Naive (7 days before)", OURS),
    ("Régression linéaire", "Linear regression", OURS),
    ("Gradient boosting", "Gradient boosting", OURS),
    ("Gradient boosting + villes (final)", "Gradient boosting + cities (final)", OURS),
    ("RTE J-1, telle que publiée", "RTE day-ahead, raw", RTE),
    ("RTE J-1, écart de mesure corrigé", "RTE day-ahead, rolling correction", RTE),
][::-1]
fig, ax = plt.subplots(figsize=(8, 3.6))
values = [results[key] for _, key, _ in bars]
ax.barh([label for label, _, _ in bars], values, color=[c for _, _, c in bars], height=0.62)
for y, v in enumerate(values):
    ax.text(v + 0.08, y, f"{v:.2f} %".replace(".", ","), va="center", fontsize=10)
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 7.2)
ax.set_xlabel("MAPE (%)")
ax.tick_params(axis="y", labelcolor=INK)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=RTE)],
          labels=["Mes modèles", "Prévision officielle RTE"], loc="lower right")
title(ax, "Erreur moyenne sur l'année de test", "MAPE, septembre 2025 à août 2026, mêmes demi-heures pour tous")
save(fig, "fig_scores.png")

# ---- 2. Two weeks around the first cold snap of the winter (our worst days)
test = pd.read_csv(RESULTS_DIR / "test_year_scored.csv", index_col=0)
test.index = pd.to_datetime(test.index, utc=True).tz_convert(TZ)
window = test.loc["2025-11-12":"2025-11-25"]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(window.index, window["load"] / 1000, color=ACTUAL, lw=2, label="Consommation réelle")
ax.plot(window.index, window["gbm_cities"] / 1000, color=OURS, lw=2, label="Mon modèle")
ax.plot(window.index, window["rte_d1_shift_by_halfhour"] / 1000, color=RTE, lw=2, label="RTE J-1 (écart corrigé)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=TZ))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m", tz=TZ))
ax.grid(axis="x", visible=False)
ax.set_ylabel("GW")
ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.0))
ax.set_ylim(top=ax.get_ylim()[1] + 4)
title(ax, "La première vague de froid de l'hiver, du 12 au 25 novembre 2025",
      "Consommation nationale par demi-heure : mon modèle sous-estime les 18, 21 et 22 novembre")
save(fig, "fig_vague_de_froid.png")

# ---- 3. Error by temperature and by type of day: ours vs RTE
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
panels = [
    ("errors_by_temperature.csv", "Selon la température moyenne du jour",
     {"< 3 °C (cold spell)": "< 3 °C", "> 22 °C (heat)": "> 22 °C"}),
    ("errors_by_daytype.csv", "Selon le type de jour",
     {"Working day": "Ouvré", "Saturday": "Samedi", "Sunday": "Dimanche", "Public holiday": "Férié",
      "Bridge day": "Pont", "Christmas week": "Noël"}),
]
for ax, (file, name, rename) in zip(axes, panels):
    table = pd.read_csv(RESULTS_DIR / file, index_col=0)
    if "Working day" in table.index:
        table = table.loc[list(rename)]
    labels = [f"{rename.get(i, i)}\n{d} j" for i, d in zip(table.index, table["days"])]
    x = range(len(table))
    for offset, column, color, label in [(-0.2, "ours", OURS, "Mon modèle"), (0.2, "rte", RTE, "RTE J-1 (écart corrigé)")]:
        ax.bar([i + offset for i in x], table[column], width=0.36, color=color, label=label)
        for i, v in zip(x, table[column]):
            ax.text(i + offset, v + 0.04, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8.5)
    ax.set_xticks(list(x), labels)
    ax.tick_params(axis="x", labelcolor=INK)
    ax.grid(axis="x", visible=False)
    ax.text(0, 1.03, name, transform=ax.transAxes, color=MUTED, fontsize=10)
axes[0].set_ylabel("MAPE (%)")
axes[0].set_title("Où se trouvent les erreurs ? Année de test", loc="left", fontsize=13, fontweight="bold", pad=26)
axes[1].legend(loc="upper left", ncols=2)
axes[0].set_ylim(0, 3.5)
save(fig, "fig_erreurs.png")
print("Figures written to results/")
