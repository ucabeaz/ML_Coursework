import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/claude/data")
OUT_DIR = Path("/home/claude/output")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GREY = "#6B7280"
PALE_BLUE = "#93C5FD"

MISSING_CUTOFF = 0.30
META = ["Ticker", "Year", "Sector", "Class"]
TARGET = "Stock_Return"


def load_year(year):
    path = DATA_DIR / f"{year}_Financial_Data.csv"
    frame = pd.read_csv(path)
    frame = frame.rename(columns={
        "Unnamed: 0": "Ticker",
        f"{year + 1} PRICE VAR [%]": TARGET,
    })
    frame["Year"] = year
    return frame


def strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(name):
    plt.savefig(FIG_DIR / name)
    plt.close()


panel = pd.concat(
    [load_year(y) for y in range(2014, 2019)],
    ignore_index=True,
)
print(f"panel: {panel.shape[0]:,} rows, {panel.shape[1]} cols")

features = [c for c in panel.columns if c not in META + [TARGET]]
print(f"features: {len(features)}")

fig, ax = plt.subplots(figsize=(6, 4))
per_year = panel.groupby("Year").size()
rects = ax.bar(per_year.index, per_year.values, color=BLUE,
               edgecolor="white", linewidth=1.2, width=0.6)
for rect, n in zip(rects, per_year.values):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 40,
            f"{n:,}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Firms")
ax.set_title("Firm-Year Observations per Year")
ax.set_xticks(per_year.index)
ax.set_ylim(0, per_year.max() * 1.12)
strip_spines(ax)
save("observations_per_year.png")

fig, ax = plt.subplots(figsize=(7, 5))
sector_n = panel["Sector"].value_counts().sort_values()
rects = ax.barh(sector_n.index, sector_n.values, color=BLUE,
                edgecolor="white", linewidth=0.8)
for rect, n in zip(rects, sector_n.values):
    ax.text(n + 30, rect.get_y() + rect.get_height() / 2,
            f"{n:,}", ha="left", va="center", fontsize=12)
ax.set_xlabel("Number of Firm-Year Observations")
ax.set_title("Observations by Sector")
strip_spines(ax)
save("sector_distribution.png")

panel = panel.dropna(subset=[TARGET])
panel = panel.loc[panel[TARGET].between(-100, 1000, inclusive="neither")]

miss_frac = panel[features].isnull().mean().sort_values(ascending=False)
drop_mask = miss_frac > MISSING_CUTOFF
drop_list = miss_frac.index[drop_mask].tolist()
keep_list = miss_frac.index[~drop_mask].tolist()
n_raw_features = len(features)
print(f"dropping {len(drop_list)} features above {MISSING_CUTOFF:.0%} missing; keeping {len(keep_list)}")

fig, ax = plt.subplots(figsize=(7, 4))
bar_cols = np.where(miss_frac.values > MISSING_CUTOFF, RED, GREEN)
ax.bar(range(len(miss_frac)), miss_frac.values * 100,
       color=bar_cols, width=1.0, linewidth=0)
ax.axhline(MISSING_CUTOFF * 100, color="black", linestyle="--",
           linewidth=1.5, label=f"{MISSING_CUTOFF:.0%} threshold")
ax.set_xlabel("Features (sorted by missingness)")
ax.set_ylabel("Missing Values (%)")
ax.set_title("Missing Data Fraction per Feature")
ax.set_xticks([])
ax.set_ylim(0, 105)
ax.legend(fontsize=13, loc="upper right")
ax.annotate(f"{drop_mask.sum()} dropped", xy=(drop_mask.sum() // 2, 65),
            fontsize=13, color=RED, ha="center", fontweight="bold")
ax.annotate(f"{(~drop_mask).sum()} retained",
            xy=(drop_mask.sum() + (~drop_mask).sum() // 2, 20),
            fontsize=13, color=GREEN, ha="center", fontweight="bold")
strip_spines(ax)
save("missing_data.png")

panel = panel.drop(columns=drop_list)
features = keep_list

panel[features] = (
    panel
    .groupby("Year")[features]
    .transform(lambda col: col.fillna(col.median()))
)
still_missing = panel[features].columns[panel[features].isnull().any()]
for col in still_missing:
    panel[col] = panel[col].fillna(panel[col].median())

bounds = panel[features].quantile([0.01, 0.99])
for col in features:
    lo, hi = bounds.loc[0.01, col], bounds.loc[0.99, col]
    panel[col] = panel[col].clip(lo, hi)

panel["Return_Direction"] = (panel[TARGET] > 0).astype(int)
pos = int((panel["Return_Direction"] == 1).sum())
neg = int((panel["Return_Direction"] == 0).sum())
total = len(panel)
print(f"positive: {pos:,} ({pos / total:.1%}); negative: {neg:,} ({neg / total:.1%})")

fig, ax = plt.subplots(figsize=(7, 4.5))
ret_vals = panel[TARGET].values
ax.hist(ret_vals, bins=100, color=BLUE, edgecolor="white",
        linewidth=0.3, alpha=0.85)
ax.axvline(0, color=RED, linestyle="--", linewidth=2, label="Zero Return")
ax.axvline(np.median(ret_vals), color=ORANGE, linestyle="-", linewidth=2,
           label=f"Median ({np.median(ret_vals):.1f}%)")
ax.set_xlabel("Annual Stock Return (%)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Annual Stock Returns")
ax.legend(fontsize=13)
strip_spines(ax)
save("return_distribution.png")

fig, ax = plt.subplots(figsize=(5, 4))
class_n = panel["Return_Direction"].value_counts().sort_index()
rects = ax.bar(["Negative\n(Return <= 0)", "Positive\n(Return > 0)"],
               class_n.values, color=[RED, GREEN],
               edgecolor="white", linewidth=1.5, width=0.5)
for rect, n in zip(rects, class_n.values):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 80,
            f"{n:,}", ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Observations")
ax.set_title("Class Distribution — Return Direction")
ax.set_ylim(0, class_n.max() * 1.15)
strip_spines(ax)
save("class_balance.png")

year_list = sorted(panel["Year"].unique())
fig, ax = plt.subplots(figsize=(7, 4.5))
box = ax.boxplot(
    [panel.loc[panel["Year"] == y, TARGET].values for y in year_list],
    labels=year_list, patch_artist=True, widths=0.5, showfliers=False,
    medianprops=dict(color="black", linewidth=2),
)
for patch in box["boxes"]:
    patch.set_facecolor(PALE_BLUE)
    patch.set_edgecolor(BLUE)
    patch.set_linewidth(1.5)
ax.axhline(0, color=RED, linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Year")
ax.set_ylabel("Annual Stock Return (%)")
ax.set_title("Stock Return Distribution by Year")
strip_spines(ax)
save("returns_by_year.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
pivot = panel.groupby(["Year", "Return_Direction"]).size().unstack(fill_value=0)
xs = np.arange(len(year_list))
width = 0.35
ax.bar(xs - width / 2, pivot[0], width, label="Negative (Return <= 0)",
       color=RED, edgecolor="white", alpha=0.85)
ax.bar(xs + width / 2, pivot[1], width, label="Positive (Return > 0)",
       color=GREEN, edgecolor="white", alpha=0.85)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Observations")
ax.set_title("Class Distribution by Year")
ax.set_xticks(xs)
ax.set_xticklabels(year_list)
ax.legend(fontsize=13)
strip_spines(ax)
save("class_balance_by_year.png")

fig, ax = plt.subplots(figsize=(7, 5.5))
ordered_sectors = panel.groupby("Sector")[TARGET].median().sort_values().index
box = ax.boxplot(
    [panel.loc[panel["Sector"] == s, TARGET].values for s in ordered_sectors],
    labels=ordered_sectors, patch_artist=True, vert=False,
    widths=0.6, showfliers=False,
    medianprops=dict(color="black", linewidth=2),
)
for patch in box["boxes"]:
    patch.set_facecolor(PALE_BLUE)
    patch.set_edgecolor(BLUE)
    patch.set_linewidth(1.2)
ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Annual Stock Return (%)")
ax.set_title("Stock Return Distribution by Sector")
ax.tick_params(axis="y", labelsize=11)
strip_spines(ax)
save("returns_by_sector.png")

CATEGORY_RULES = [
    ("Profitability", ("margin", "profit", "return", "roe", "roa", "roic",
                       "ebit", "ebitda", "income", "earning", "eps", "net income")),
    ("Leverage",      ("debt", "leverage", "equity", "liabilit", "interest",
                       "coverage", "capital", "solvency")),
    ("Liquidity",     ("cash", "current", "quick", "liquid", "working capital",
                       "receivable", "payable", "inventory", "turnover", "days")),
    ("Valuation",     ("price", "book", "pe ", "p/e", "market", "enterprise",
                       "ev", "dividend", "yield", "valuation", "fair value",
                       "sales ratio")),
    ("Growth",        ("growth", "change")),
]


def classify(name):
    lowered = name.lower()
    for label, tokens in CATEGORY_RULES:
        if any(tok in lowered for tok in tokens):
            return label
    return "Other"


cat_lookup = {c: classify(c) for c in features}
cat_n = pd.Series(cat_lookup).value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
palette = [BLUE, RED, GREEN, ORANGE, PURPLE, GREY][:len(cat_n)]
rects = ax.barh(cat_n.index, cat_n.values, color=palette,
                edgecolor="white", linewidth=1)
for rect, n in zip(rects, cat_n.values):
    ax.text(n + 1, rect.get_y() + rect.get_height() / 2, str(n),
            ha="left", va="center", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Features")
ax.set_title("Feature Categories")
strip_spines(ax)
save("feature_categories.png")

correlations = panel[features].corrwith(panel[TARGET]).dropna()
extremes = pd.concat([
    correlations.nsmallest(10),
    correlations.nlargest(10),
]).sort_values()

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(
    [n if len(n) <= 35 else n[:35] + "..." for n in extremes.index],
    extremes.values,
    color=[RED if v < 0 else GREEN for v in extremes.values],
    edgecolor="white", linewidth=0.8,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Stock Return")
ax.set_title("Top 20 Features by Correlation with Return")
ax.tick_params(axis="y", labelsize=10)
strip_spines(ax)
save("top_correlations.png")

cleaned_path = OUT_DIR / "cleaned_dataset.csv"
panel.to_csv(cleaned_path, index=False)

print(f"\nfinal: {panel.shape[0]:,} x {panel.shape[1]}")
print(f"features retained: {len(features)} (of {n_raw_features})")
print(f"sectors: {panel['Sector'].nunique()}")
print(f"saved dataset -> {cleaned_path}")
print(f"saved figures -> {FIG_DIR}")
