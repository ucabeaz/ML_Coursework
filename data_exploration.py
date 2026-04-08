import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/claude/output")
FIG_DIR = OUT_DIR / "exploration_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
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
PALE_BLUE = "#93C5FD"

NON_FEATURE = {"Ticker", "Year", "Sector", "Class", "Stock_Return", "Return_Direction"}
KENDALL_SAMPLE = 8000
SEED = 42


def strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(name):
    plt.savefig(FIG_DIR / name)
    plt.close()


def shorten(name, n=28):
    return name if len(name) <= n else name[: n - 2] + ".."


df = pd.read_csv(OUT_DIR / "cleaned_dataset.csv")
features = [c for c in df.columns if c not in NON_FEATURE]
print(f"loaded {df.shape[0]:,} rows, {len(features)} features")

ret = df["Stock_Return"]
print(
    f"return: mean={ret.mean():.2f} median={ret.median():.2f} "
    f"std={ret.std():.2f} skew={ret.skew():.2f} kurt={ret.kurtosis():.2f}"
)
print(f"IQR: [{ret.quantile(0.25):.2f}, {ret.quantile(0.75):.2f}]")

candidate_summary = [
    "Revenue", "Gross Profit", "Net Income", "EPS", "EPS Diluted",
    "Operating Income", "EBITDA", "Free Cash Flow per Share",
    "returnOnEquity", "returnOnAssets", "ROIC", "Profit Margin",
    "Debt to Equity", "Interest Coverage", "Current Ratio",
    "priceEarningsRatio", "priceToBookRatio", "priceSalesRatio",
    "Revenue Growth", "Asset Growth",
]
summary_cols = [c for c in candidate_summary if c in features]
summary = (
    df[summary_cols + ["Stock_Return"]]
    .describe()
    .T[["mean", "50%", "std", "min", "max"]]
    .round(2)
)
summary.columns = ["Mean", "Median", "Std", "Min", "Max"]
summary.to_csv(FIG_DIR / "descriptive_statistics.csv")

year_palette = [BLUE, RED, GREEN, ORANGE, PURPLE]
fig, ax = plt.subplots(figsize=(7, 4.5))
for idx, yr in enumerate(sorted(df["Year"].unique())):
    series = df.loc[df["Year"] == yr, "Stock_Return"]
    series.plot.kde(ax=ax, label=f"{yr + 1} Returns",
                    color=year_palette[idx], linewidth=2)
ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlim(-150, 250)
ax.set_xlabel("Annual Stock Return (%)")
ax.set_ylabel("Density")
ax.set_title("Return Distribution by Year (KDE)")
ax.legend(fontsize=11)
strip_spines(ax)
save("kde_returns_by_year.png")

ordered_sectors = df.groupby("Sector")["Stock_Return"].median().sort_values().index
fig, ax = plt.subplots(figsize=(7, 5.5))
box = ax.boxplot(
    [df.loc[df["Sector"] == s, "Stock_Return"].values for s in ordered_sectors],
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

years_sorted = sorted(df["Year"].unique())
class_pivot = (
    df.groupby(["Year", "Return_Direction"]).size().unstack(fill_value=0)
)
class_pct = class_pivot.div(class_pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
xs = np.arange(len(years_sorted))
width = 0.35
neg_bars = ax.bar(xs - width / 2, class_pct[0], width,
                  label="Negative (Return <= 0)",
                  color=RED, edgecolor="white", alpha=0.85)
pos_bars = ax.bar(xs + width / 2, class_pct[1], width,
                  label="Positive (Return > 0)",
                  color=GREEN, edgecolor="white", alpha=0.85)
for group in (neg_bars, pos_bars):
    for bar in group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
ax.set_xlabel("Return Year")
ax.set_ylabel("Proportion (%)")
ax.set_title("Class Balance by Year")
ax.set_xticks(xs)
ax.set_xticklabels([y + 1 for y in years_sorted])
ax.set_ylim(0, 85)
ax.legend(fontsize=11)
strip_spines(ax)
save("class_balance_by_year.png")

print("computing correlations...")
pearson = df[features].corrwith(df["Stock_Return"], method="pearson")
spearman = df[features].corrwith(df["Stock_Return"], method="spearman")

rng = np.random.default_rng(SEED)
sample_size = min(KENDALL_SAMPLE, len(df))
sample_idx = rng.choice(df.index.values, size=sample_size, replace=False)
sample = df.loc[sample_idx]
kendall = sample[features].corrwith(sample["Stock_Return"], method="kendall")

corr_table = (
    pd.DataFrame({"Pearson": pearson, "Spearman": spearman, "Kendall": kendall})
    .dropna()
)
corr_table["AbsP"] = corr_table["Pearson"].abs()
corr_table = corr_table.sort_values("AbsP", ascending=False)
corr_table[["Pearson", "Spearman", "Kendall"]].to_csv(
    FIG_DIR / "feature_return_correlations.csv"
)

TOP_N = 15
top_block = corr_table.head(TOP_N)[["Pearson", "Spearman", "Kendall"]]

fig, ax = plt.subplots(figsize=(7, 6))
xs = np.arange(TOP_N)
w = 0.25
ax.barh(xs + w, top_block["Pearson"], w, label="Pearson",
        color=BLUE, edgecolor="white", alpha=0.9)
ax.barh(xs, top_block["Spearman"], w, label="Spearman",
        color=ORANGE, edgecolor="white", alpha=0.9)
ax.barh(xs - w, top_block["Kendall"], w, label="Kendall",
        color=GREEN, edgecolor="white", alpha=0.9)
ax.set_yticks(xs)
ax.set_yticklabels([shorten(n) for n in top_block.index], fontsize=10)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Correlation with Stock Return")
ax.set_title("Top 15 Features: Three Correlation Measures")
ax.legend(fontsize=11, loc="lower right")
ax.invert_yaxis()
strip_spines(ax)
save("correlation_comparison_top15.png")

top10 = corr_table.head(10)[["Pearson", "Spearman", "Kendall"]].copy()
top10.index = [shorten(n, 30) for n in top10.index]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(top10, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            linewidths=0.8, linecolor="white",
            cbar_kws={"label": "Correlation Coefficient", "shrink": 0.8},
            ax=ax, annot_kws={"size": 12})
ax.set_title("Top 10 Features: Correlation with Return")
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=12)
save("correlation_heatmap_top10.png")

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(corr_table["Pearson"], corr_table["Spearman"], alpha=0.5, s=25,
           color=BLUE, edgecolor="white", linewidth=0.3)
extent = max(corr_table["Pearson"].abs().max(),
             corr_table["Spearman"].abs().max()) * 1.1
ax.plot([-extent, extent], [-extent, extent], "k--",
        linewidth=1, alpha=0.5, label="y = x")
divergence = (corr_table["Pearson"] - corr_table["Spearman"]).abs()
for feat in divergence.nlargest(5).index:
    p, s = corr_table.loc[feat, "Pearson"], corr_table.loc[feat, "Spearman"]
    ax.annotate(shorten(feat, 20), (p, s), fontsize=8, alpha=0.8,
                xytext=(5, 5), textcoords="offset points")
ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Pearson Correlation")
ax.set_ylabel("Spearman Correlation")
ax.set_title("Pearson vs Spearman: All Features")
ax.legend(fontsize=11)
strip_spines(ax)
save("pearson_vs_spearman.png")

top20_feats = corr_table.head(20).index.tolist()
top20_corr = df[top20_feats].corr(method="pearson")

fig, ax = plt.subplots(figsize=(10, 9))
labels_short = [shorten(n, 22) for n in top20_corr.columns]
mask = np.triu(np.ones_like(top20_corr, dtype=bool), k=1)
sns.heatmap(top20_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, linecolor="white", square=True,
            xticklabels=labels_short, yticklabels=labels_short,
            cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
            ax=ax, annot_kws={"size": 8}, vmin=-1, vmax=1)
ax.set_title("Cross-Correlation: Top 20 Predictive Features", fontsize=16, pad=15)
ax.tick_params(axis="both", labelsize=9)
plt.xticks(rotation=45, ha="right")
save("cross_correlation_heatmap.png")

top20_pairs = []
for i in range(len(top20_corr)):
    for j in range(i + 1, len(top20_corr)):
        r = top20_corr.iat[i, j]
        if abs(r) > 0.80:
            top20_pairs.append((top20_corr.index[i], top20_corr.columns[j], r))
top20_pairs.sort(key=lambda t: abs(t[2]), reverse=True)

full_corr = df[features].corr(method="pearson").values
iu, ju = np.triu_indices(full_corr.shape[0], k=1)
pairwise = full_corr[iu, ju]
n_above_90 = int((np.abs(pairwise) > 0.90).sum())
n_above_95 = int((np.abs(pairwise) > 0.95).sum())
n_above_99 = int((np.abs(pairwise) > 0.99).sum())
print(f"|r| > 0.90: {n_above_90}, > 0.95: {n_above_95}, > 0.99: {n_above_99}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(pairwise, bins=100, color=BLUE, edgecolor="white",
        linewidth=0.3, alpha=0.85)
ax.axvline(0.90, color=RED, linestyle="--", linewidth=2,
           label=f"|r| = 0.90 ({n_above_90} pairs)")
ax.axvline(-0.90, color=RED, linestyle="--", linewidth=2)
ax.set_xlabel("Pairwise Pearson Correlation")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Pairwise Feature Correlations")
ax.legend(fontsize=12)
strip_spines(ax)
save("pairwise_correlation_distribution.png")

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


cat_series = pd.Series({c: classify(c) for c in features})
cat_frame = pd.DataFrame({
    "Category": cat_series,
    "AbsPearson": corr_table["Pearson"].abs(),
    "AbsSpearman": corr_table["Spearman"].abs(),
}).dropna()
cat_means = (
    cat_frame.groupby("Category")[["AbsPearson", "AbsSpearman"]]
    .mean()
    .sort_values("AbsPearson")
)

fig, ax = plt.subplots(figsize=(7, 4.5))
xs = np.arange(len(cat_means))
w = 0.35
ax.barh(xs - w / 2, cat_means["AbsPearson"], w,
        label="Pearson", color=BLUE, edgecolor="white")
ax.barh(xs + w / 2, cat_means["AbsSpearman"], w,
        label="Spearman", color=ORANGE, edgecolor="white")
ax.set_yticks(xs)
ax.set_yticklabels(cat_means.index, fontsize=12)
ax.set_xlabel("Mean |Correlation| with Stock Return")
ax.set_title("Predictive Strength by Feature Category")
ax.legend(fontsize=11)
strip_spines(ax)
save("category_correlation_strength.png")

top4 = corr_table.head(4).index.tolist()
sample = df.sample(n=3000, random_state=SEED)
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, feat in zip(axes.flat, top4):
    ax.scatter(sample[feat], sample["Stock_Return"], alpha=0.15, s=8,
               color=BLUE, edgecolor="none")
    try:
        bins = pd.qcut(sample[feat], q=20, duplicates="drop")
        means = sample.groupby(bins)["Stock_Return"].mean()
        centres = sample.groupby(bins)[feat].mean()
        ax.plot(centres, means, color=RED, linewidth=2.5,
                label="Binned Mean", zorder=5)
    except Exception:
        pass
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(shorten(feat, 25), fontsize=12)
    ax.set_ylabel("Stock Return (%)", fontsize=12)
    strip_spines(ax)
axes.flat[0].legend(fontsize=10, loc="upper left")
fig.suptitle("Top Features vs Stock Return (with Binned Means)",
             fontsize=16, y=1.01)
plt.tight_layout()
save("nonlinearity_scatter.png")

ranks = pd.DataFrame({
    "P": corr_table["Pearson"].abs().rank(ascending=False),
    "S": corr_table["Spearman"].abs().rank(ascending=False),
    "K": corr_table["Kendall"].abs().rank(ascending=False),
})
print(f"rank agreement P-S: {ranks['P'].corr(ranks['S'], method='spearman'):.3f}")
print(f"rank agreement P-K: {ranks['P'].corr(ranks['K'], method='spearman'):.3f}")
print(f"rank agreement S-K: {ranks['S'].corr(ranks['K'], method='spearman'):.3f}")
print(f"figures saved -> {FIG_DIR}")
