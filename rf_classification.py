from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.dummy import DummyClassifier

BASE_DIR = Path(os.environ.get("COMP0050_BASE",
                               Path.home() / "Desktop" / "Machine Learning"))
DATA_PATH = BASE_DIR / "cleaned_dataset.csv"
OUT_DIR = BASE_DIR / "rf_classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "figure.figsize": (7, 4.5), "font.size": 10})
POS, NEG, NEUTRAL = "#2a9d8f", "#e76f51", "#264653"

NON_FEATURES = ["Ticker", "Sector", "Stock_Return",
                "Return_Direction", "Year", "Class"]
TRAIN_YEARS, TEST_YEAR = [2014, 2015, 2016, 2017], 2018
TARGET_COL = "Class"
SEED = 0

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

sec_train = pd.get_dummies(train_df["Sector"], prefix="sec", dummy_na=True)
sec_test  = pd.get_dummies(test_df["Sector"],  prefix="sec", dummy_na=True)
sec_test  = sec_test.reindex(columns=sec_train.columns, fill_value=0)

X_train = pd.concat([X_train_num, sec_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([X_test_num,  sec_test.reset_index(drop=True)],  axis=1)
y_train = train_df[TARGET_COL].astype(int).values
y_test  = test_df[TARGET_COL].astype(int).values
feature_names = X_train.columns.tolist()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train pos rate: {y_train.mean():.3f} | Test pos rate: {y_test.mean():.3f}")

tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators":     [100, 300],
    "max_depth":        [5, 10, None],
    "min_samples_split":[2, 5, 10],
}
grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced",
                           random_state=SEED, n_jobs=-1),
    param_grid, scoring="f1_macro", cv=tscv, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"CV macro-F1: {grid.best_score_:.4f}")

rf = grid.best_estimator_
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

def metrics_row(name, y_true, y_pred, y_proba=None):
    return {
        "Model": name,
        "Acc":     accuracy_score(y_true, y_pred),
        "AUC":     roc_auc_score(y_true, y_proba) if y_proba is not None else 0.5,
        "Prec":    precision_score(y_true, y_pred, zero_division=0),
        "Rec":     recall_score(y_true, y_pred, zero_division=0),
        "F1":      f1_score(y_true, y_pred, zero_division=0),
        "MacroF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

maj = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
rng = DummyClassifier(strategy="stratified", random_state=SEED).fit(X_train, y_train)

results = pd.DataFrame([
    metrics_row("Majority Class",     y_test, maj.predict(X_test)),
    metrics_row("Stratified Random",  y_test, rng.predict(X_test), rng.predict_proba(X_test)[:, 1]),
    metrics_row("Random Forest",      y_test, y_pred, y_proba),
])
print(results.to_string(index=False))

perm = permutation_importance(rf, X_test, y_test,
                              n_repeats=10, random_state=SEED,
                              scoring="roc_auc", n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
top = imp.head(20)

fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(top.index[::-1], top.values[::-1], color=POS)
ax.set(title="Random Forest Classifier — Top 20 Permutation Importances",
       xlabel="Mean drop in AUC when feature shuffled")
fig.tight_layout()
fig.savefig(OUT_DIR / "rf_class_importance.png"); plt.close(fig)

with open(OUT_DIR / "metrics.txt", "w") as f:
    f.write("Random Forest Classification — results\n")
    f.write("========================================\n")
    f.write(f"Best params       : {grid.best_params_}\n")
    f.write(f"CV macro-F1       : {grid.best_score_:.4f}\n")
    f.write(f"Train pos rate    : {y_train.mean():.4f}\n")
    f.write(f"Test  pos rate    : {y_test.mean():.4f}\n\n")
    f.write(results.to_string(index=False))
    f.write("\n\nConfusion matrix (RF):\n")
    f.write(np.array2string(confusion_matrix(y_test, y_pred)))
    f.write("\n\nTop 20 features by permutation importance:\n")
    f.write(imp.head(20).to_string())
print(f"Done. Outputs written to {OUT_DIR}")
