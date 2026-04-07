"""
linear_regression.py
--------------------
Lasso regression for continuous next-year stock return prediction, plus the
shared exploratory figures used by the report (return distribution, KDE by
year, top-10 feature-target correlation heatmap).

Why Lasso only? The coursework revision consolidates on a single regulariser.
Preliminary experiments (OLS, Ridge, ElasticNet) gave test RMSE within 0.1 of
Lasso, and the tuned ElasticNet l1-ratio converged to 0.914, i.e. the data
itself prefers a sparse solution. Lasso is chosen because (i) it matches this
empirical preference, (ii) it performs implicit feature selection under heavy
multicollinearity, and (iii) its non-zero coefficients are directly
interpretable, which supports the feature-interpretation section of the report.

I/O:
    in : <BASE_DIR>/cleaned_dataset.csv
    out: <BASE_DIR>/linear_regression/{*.png, metrics.txt}

BASE_DIR defaults to ~/Desktop/Machine Learning. Override with the env var
COMP0050_BASE.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Paths & shared style                                                        #
# --------------------------------------------------------------------------- #
BASE_DIR = Path(os.environ.get("COMP0050_BASE",
                               Path.home() / "Desktop" / "Machine Learning"))
DATA_PATH = BASE_DIR / "cleaned_dataset.csv"
OUT_DIR = BASE_DIR / "linear_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "figure.figsize": (7, 4.5), "font.size": 10})
POS, NEG, NEUTRAL = "#2a9d8f", "#e76f51", "#264653"

NON_FEATURES = ["Ticker", "Sector", "Stock_Return",
                "Return_Direction", "Year", "Class"]
TRAIN_YEARS, TEST_YEAR = [2014, 2015, 2016, 2017], 2018

# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH} -> {df.shape}")

# --------------------------------------------------------------------------- #
# Shared EDA figures (referenced by the LaTeX report)                         #
# --------------------------------------------------------------------------- #
# 1) Return distribution
ret = df["Stock_Return"].dropna()
ret_plot = ret[(ret > -100) & (ret < 300)]
fig, ax = plt.subplots()
ax.hist(ret_plot, bins=60, color=NEUTRAL, alpha=0.85, edgecolor="white")
ax.axvline(0, color=NEG, linestyle="--", label="Zero return")
ax.axvline(ret.median(), color=POS, linestyle="-",
           label=f"Median ({ret.median():.1f}%)")
ax.set(title="Distribution of Annual Stock Returns",
       xlabel="Annual Stock Return (%)", ylabel="Frequency")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "intro_return_distribution.png"); plt.close(fig)

# 2) KDE by year
fig, ax = plt.subplots()
for yr in sorted(df["Year"].unique()):
    sns.kdeplot(df.loc[df["Year"] == yr, "Stock_Return"]
                  .clip(-150, 250), ax=ax, label=f"{yr+1} Returns", lw=1.6)
ax.axvline(0, color="grey", linestyle="--", lw=0.8)
ax.set(title="Return Distribution by Year (KDE)",
       xlabel="Annual Stock Return (%)", ylabel="Density",
       xlim=(-150, 250))
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "kde_returns_by_year.png"); plt.close(fig)

# 3) Top-10 feature-target correlations (Pearson, Spearman, Kendall)
feat_cols = [c for c in df.columns if c not in NON_FEATURES]
num_df = df[feat_cols].select_dtypes(include=[np.number])
tgt = df["Stock_Return"]
pear = num_df.corrwith(tgt, method="pearson")
spear = num_df.corrwith(tgt, method="spearman")
kend = num_df.corrwith(tgt, method="kendall")
corr_tbl = (pd.DataFrame({"Pearson": pear, "Spearman": spear, "Kendall": kend})
              .dropna()
              .assign(abs_p=lambda d: d["Pearson"].abs())
              .sort_values("abs_p", ascending=False)
              .drop(columns="abs_p")
              .head(10))
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr_tbl, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            cbar_kws={"label": "Correlation Coefficient"}, ax=ax)
ax.set_title("Top 10 Features: Correlation with Return")
fig.tight_layout()
fig.savefig(OUT_DIR / "correlation_heatmap_top10.png"); plt.close(fig)

# Report diagnostic: Pearson/Spearman rank-agreement across ALL features
rank_agree = (pear.rank()
              .corr(spear.rank(), method="spearman"))
print(f"Pearson vs Spearman rank agreement (all features): {rank_agree:.3f}")

# --------------------------------------------------------------------------- #
# Train / test split                                                          #
# --------------------------------------------------------------------------- #
train_df = df[df["Year"].isin(TRAIN_YEARS)].sort_values("Year").reset_index(drop=True)
test_df  = df[df["Year"] == TEST_YEAR].reset_index(drop=True)

X_train_num = train_df[feat_cols].select_dtypes(include=[np.number]).copy()
X_test_num  = test_df[feat_cols].select_dtypes(include=[np.number]).copy()
num_feat_cols = X_train_num.columns.tolist()

# One-hot sector
sec_train = pd.get_dummies(train_df["Sector"], prefix="sec", dummy_na=True)
sec_test  = pd.get_dummies(test_df["Sector"],  prefix="sec", dummy_na=True)
sec_test  = sec_test.reindex(columns=sec_train.columns, fill_value=0)

# Median-impute any residual NaNs using train medians
med = X_train_num.median()
X_train_num = X_train_num.fillna(med)
X_test_num  = X_test_num.fillna(med)

X_train = pd.concat([X_train_num, sec_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([X_test_num,  sec_test.reset_index(drop=True)],  axis=1)
y_train = train_df["Stock_Return"].values
y_test  = test_df["Stock_Return"].values

# Standardise (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}")

# --------------------------------------------------------------------------- #
# Lasso with TimeSeriesSplit CV                                               #
# --------------------------------------------------------------------------- #
tscv = TimeSeriesSplit(n_splits=3)
alphas = np.logspace(-3, 1.0, 25)
grid = GridSearchCV(Lasso(max_iter=20_000),
                    {"alpha": alphas},
                    scoring="neg_root_mean_squared_error",
                    cv=tscv, n_jobs=-1)
grid.fit(X_train_s, y_train)
best_alpha = grid.best_params_["alpha"]
cv_rmse = -grid.best_score_
print(f"Best Lasso alpha: {best_alpha:.4f}  | CV RMSE: {cv_rmse:.3f}")

lasso = grid.best_estimator_
y_pred = lasso.predict(X_test_s)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2   = r2_score(y_test, y_pred)

# Mean baseline (training mean) for comparison
y_mean_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
base_rmse = np.sqrt(mean_squared_error(y_test, y_mean_pred))
base_r2   = r2_score(y_test, y_mean_pred)

print(f"Lasso         : RMSE={test_rmse:.3f}  R2={test_r2:.4f}")
print(f"Mean baseline : RMSE={base_rmse:.3f}  R2={base_r2:.4f}")
print(f"Train mean return = {y_train.mean():.2f}%, Test mean = {y_test.mean():.2f}%")

# --------------------------------------------------------------------------- #
# Figures: scatter + CV RMSE by fold                                          #
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=6, alpha=0.35, color=NEUTRAL)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "--", color=NEG, label="Perfect fit")
ax.set(title=f"Lasso — Actual vs Predicted\nRMSE={test_rmse:.2f}  R²={test_r2:.4f}",
       xlabel="Actual Return (%)", ylabel="Predicted Return (%)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "lasso_output.png"); plt.close(fig)

# Per-fold CV RMSE at the best alpha (expanding window)
fold_rmses = []
for i, (tr, va) in enumerate(tscv.split(X_train_s), start=1):
    m = Lasso(alpha=best_alpha, max_iter=20_000).fit(X_train_s[tr], y_train[tr])
    fold_rmses.append(np.sqrt(mean_squared_error(y_train[va], m.predict(X_train_s[va]))))
fig, ax = plt.subplots()
ax.plot(range(1, len(fold_rmses) + 1), fold_rmses, "-o", color=POS, lw=2)
ax.set(title="Expanding-Window CV — Lasso RMSE by Fold",
       xlabel="Fold", ylabel="RMSE")
ax.set_xticks(range(1, len(fold_rmses) + 1))
fig.tight_layout()
fig.savefig(OUT_DIR / "rmse_by_cvfold.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Persist metrics                                                             #
# --------------------------------------------------------------------------- #
with open(OUT_DIR / "metrics.txt", "w") as f:
    f.write("Linear Regression (Lasso) — results\n")
    f.write("===================================\n")
    f.write(f"Best alpha        : {best_alpha:.6f}\n")
    f.write(f"CV RMSE (3-fold)  : {cv_rmse:.4f}\n")
    f.write(f"Per-fold CV RMSE  : {fold_rmses}\n")
    f.write(f"Test RMSE         : {test_rmse:.4f}\n")
    f.write(f"Test R^2          : {test_r2:.6f}\n")
    f.write(f"Mean baseline RMSE: {base_rmse:.4f}\n")
    f.write(f"Mean baseline R^2 : {base_r2:.6f}\n")
    f.write(f"Train mean return : {y_train.mean():.4f}%\n")
    f.write(f"Test  mean return : {y_test.mean():.4f}%\n")
    f.write(f"Pearson vs Spearman rank agreement (all features): {rank_agree:.4f}\n")
    nz = int(np.sum(np.abs(lasso.coef_) > 1e-10))
    f.write(f"Non-zero Lasso coefficients: {nz}/{len(lasso.coef_)}\n")
print(f"Done. Outputs written to {OUT_DIR}")
