"""
rf_regression.py
----------------
Random Forest regressor for continuous next-year stock return prediction.

Uses permutation importance (not impurity-based) because impurity importance
is biased toward high-cardinality/continuous features and is known to be
unreliable under strong multicollinearity — exactly the setting here.

I/O:
    in : <BASE_DIR>/cleaned_dataset.csv
    out: <BASE_DIR>/rf_regression/{rf_feature_importance.png, metrics.txt}
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
BASE_DIR = Path(os.environ.get("COMP0050_BASE",
                               Path.home() / "Desktop" / "Machine Learning"))
DATA_PATH = BASE_DIR / "cleaned_dataset.csv"
OUT_DIR = BASE_DIR / "rf_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "figure.figsize": (7, 4.5), "font.size": 10})
POS, NEG, NEUTRAL = "#2a9d8f", "#e76f51", "#264653"

NON_FEATURES = ["Ticker", "Sector", "Stock_Return",
                "Return_Direction", "Year", "Class"]
TRAIN_YEARS, TEST_YEAR = [2014, 2015, 2016, 2017], 2018
SEED = 0

# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH} -> {df.shape}")

feat_cols = [c for c in df.columns if c not in NON_FEATURES]
train_df = df[df["Year"].isin(TRAIN_YEARS)].sort_values("Year").reset_index(drop=True)
test_df  = df[df["Year"] == TEST_YEAR].reset_index(drop=True)

X_train_num = train_df[feat_cols].select_dtypes(include=[np.number]).copy()
X_test_num  = test_df[feat_cols].select_dtypes(include=[np.number]).copy()
med = X_train_num.median()
X_train_num = X_train_num.fillna(med)
X_test_num  = X_test_num.fillna(med)

# One-hot sector (trees handle raw dummies fine)
sec_train = pd.get_dummies(train_df["Sector"], prefix="sec", dummy_na=True)
sec_test  = pd.get_dummies(test_df["Sector"],  prefix="sec", dummy_na=True)
sec_test  = sec_test.reindex(columns=sec_train.columns, fill_value=0)

X_train = pd.concat([X_train_num, sec_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([X_test_num,  sec_test.reset_index(drop=True)],  axis=1)
y_train = train_df["Stock_Return"].values
y_test  = test_df["Stock_Return"].values
feature_names = X_train.columns.tolist()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --------------------------------------------------------------------------- #
# Grid search with expanding-window CV                                        #
# --------------------------------------------------------------------------- #
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators":     [100, 300],
    "max_depth":        [5, 10, None],
    "min_samples_split":[2, 5, 10],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=tscv, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"CV RMSE    : {-grid.best_score_:.3f}")

rf = grid.best_estimator_
y_pred = rf.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2   = r2_score(y_test, y_pred)
base_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean(), dtype=float)))
base_r2   = r2_score(y_test, np.full_like(y_test, y_train.mean(), dtype=float))
print(f"RF   : RMSE={test_rmse:.3f}  R2={test_r2:.4f}")
print(f"Mean : RMSE={base_rmse:.3f}  R2={base_r2:.4f}")

# --------------------------------------------------------------------------- #
# Permutation importance (test set, 10 repeats)                               #
# --------------------------------------------------------------------------- #
perm = permutation_importance(rf, X_test, y_test,
                              n_repeats=10, random_state=SEED,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
top = imp.head(15)

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top.index[::-1], top.values[::-1], color=POS)
ax.set(title="Random Forest Regressor — Top 15 Permutation Importances",
       xlabel="Mean increase in RMSE when feature shuffled")
fig.tight_layout()
fig.savefig(OUT_DIR / "rf_feature_importance.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
with open(OUT_DIR / "metrics.txt", "w") as f:
    f.write("Random Forest Regression — results\n")
    f.write("===================================\n")
    f.write(f"Best params      : {grid.best_params_}\n")
    f.write(f"CV RMSE (3-fold) : {-grid.best_score_:.4f}\n")
    f.write(f"Test RMSE        : {test_rmse:.4f}\n")
    f.write(f"Test R^2         : {test_r2:.6f}\n")
    f.write(f"Baseline RMSE    : {base_rmse:.4f}\n")
    f.write(f"Baseline R^2     : {base_r2:.6f}\n")
    f.write(f"RF mean prediction: {y_pred.mean():.4f}\n")
    f.write(f"Test actual mean  : {y_test.mean():.4f}\n\n")
    f.write("Top 20 features by permutation importance:\n")
    f.write(imp.head(20).to_string())
print(f"Done. Outputs written to {OUT_DIR}")
