"""
linear_classification.py
------------------------
L1-penalised (Lasso) logistic regression for predicting the direction of
next-year stock returns. Consolidated to a single regulariser to match the
regression pipeline and to support interpretable coefficient-level analysis.

Why L1? (i) Mirrors the linear regression choice, (ii) the high
multicollinearity in the feature set (131 pairs with |r|>0.9) makes implicit
feature selection valuable, (iii) L2 and unregularised variants were
evaluated in preliminary work and gave AUCs within ~0.05 of L1, so the
qualitative conclusions are unchanged and L1 offers materially better
interpretability.

I/O:
    in : <BASE_DIR>/cleaned_dataset.csv
    out: <BASE_DIR>/linear_classification/{lasso_coefficients.png, metrics.txt}
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.dummy import DummyClassifier

# --------------------------------------------------------------------------- #
BASE_DIR = Path(os.environ.get("COMP0050_BASE",
                               Path.home() / "Desktop" / "Machine Learning"))
DATA_PATH = BASE_DIR / "cleaned_dataset.csv"
OUT_DIR = BASE_DIR / "linear_classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "figure.figsize": (7, 4.5), "font.size": 10})
POS, NEG, NEUTRAL = "#2a9d8f", "#e76f51", "#264653"

NON_FEATURES = ["Ticker", "Sector", "Stock_Return",
                "Return_Direction", "Year", "Class"]
TRAIN_YEARS, TEST_YEAR = [2014, 2015, 2016, 2017], 2018
TARGET_COL = "Class"  # binary direction label

# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH} -> {df.shape}")

feat_cols = [c for c in df.columns if c not in NON_FEATURES]
train_df = df[df["Year"].isin(TRAIN_YEARS)].sort_values("Year").reset_index(drop=True)
test_df  = df[df["Year"] == TEST_YEAR].reset_index(drop=True)

X_train_num = train_df[feat_cols].select_dtypes(include=[np.number]).copy()
X_test_num  = test_df[feat_cols].select_dtypes(include=[np.number]).copy()

# Median-impute using train medians
med = X_train_num.median()
X_train_num = X_train_num.fillna(med)
X_test_num  = X_test_num.fillna(med)

# --------------------------------------------------------------------------- #
# Correlation pruning: drop one of each pair with |r| > 0.80                  #
# (L1 solver was unstable without this in the original pipeline)              #
# --------------------------------------------------------------------------- #
corr = X_train_num.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.80)]
X_train_num = X_train_num.drop(columns=drop_cols)
X_test_num  = X_test_num.drop(columns=drop_cols)
print(f"After |r|>0.80 pruning: {X_train_num.shape[1]} features (dropped {len(drop_cols)})")

# One-hot sector
sec_train = pd.get_dummies(train_df["Sector"], prefix="sec", dummy_na=True)
sec_test  = pd.get_dummies(test_df["Sector"],  prefix="sec", dummy_na=True)
sec_test  = sec_test.reindex(columns=sec_train.columns, fill_value=0)

X_train = pd.concat([X_train_num, sec_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([X_test_num,  sec_test.reset_index(drop=True)],  axis=1)
feature_names = X_train.columns.tolist()
y_train = train_df[TARGET_COL].astype(int).values
y_test  = test_df[TARGET_COL].astype(int).values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}")
print(f"Train positive rate: {y_train.mean():.3f}")
print(f"Test  positive rate: {y_test.mean():.3f}")

# --------------------------------------------------------------------------- #
# Grid search C for L1 logistic (TimeSeriesSplit, AUC)                        #
# --------------------------------------------------------------------------- #
tscv = TimeSeriesSplit(n_splits=3)
Cs = np.logspace(-4, 1, 12)
grid = GridSearchCV(
    LogisticRegression(penalty="l1", solver="saga",
                       class_weight="balanced",
                       max_iter=5_000, tol=1e-3),
    {"C": Cs}, scoring="roc_auc", cv=tscv, n_jobs=-1)
grid.fit(X_train_s, y_train)
best_C = grid.best_params_["C"]
print(f"Best C: {best_C:.5f}  | CV AUC: {grid.best_score_:.4f}")

clf = grid.best_estimator_
y_pred  = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1]

def metrics_row(name, y_true, y_pred, y_proba=None):
    return {
        "Model": name,
        "Acc":      accuracy_score(y_true, y_pred),
        "AUC":      roc_auc_score(y_true, y_proba) if y_proba is not None else 0.5,
        "Prec":     precision_score(y_true, y_pred, zero_division=0),
        "Rec":      recall_score(y_true, y_pred, zero_division=0),
        "F1":       f1_score(y_true, y_pred, zero_division=0),
        "MacroF1":  f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

# Baselines
maj = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)
maj_pred = maj.predict(X_test_s)
rng = DummyClassifier(strategy="stratified", random_state=0).fit(X_train_s, y_train)
rng_pred = rng.predict(X_test_s)
rng_proba = rng.predict_proba(X_test_s)[:, 1]

results = pd.DataFrame([
    metrics_row("Majority Class", y_test, maj_pred),
    metrics_row("Stratified Random", y_test, rng_pred, rng_proba),
    metrics_row("L1 Logistic (Lasso)", y_test, y_pred, y_proba),
])
print(results.to_string(index=False))

# --------------------------------------------------------------------------- #
# Top non-zero coefficients figure                                            #
# --------------------------------------------------------------------------- #
coefs = pd.Series(clf.coef_.ravel(), index=feature_names)
nz = coefs[coefs.abs() > 1e-10].sort_values(key=np.abs, ascending=False).head(20)
colors = [POS if v > 0 else NEG for v in nz.values]

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(nz.index[::-1], nz.values[::-1], color=colors[::-1])
ax.axvline(0, color="black", lw=0.6)
ax.set(title=f"L1 Logistic — Top Features ({int((coefs.abs()>1e-10).sum())} non-zero of {len(coefs)})",
       xlabel="Coefficient Value")
fig.tight_layout()
fig.savefig(OUT_DIR / "lasso_coefficients.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
with open(OUT_DIR / "metrics.txt", "w") as f:
    f.write("Linear Classification (L1 Logistic / Lasso) — results\n")
    f.write("======================================================\n")
    f.write(f"Best C           : {best_C:.6f}\n")
    f.write(f"CV AUC (3-fold)  : {grid.best_score_:.4f}\n")
    f.write(f"Train pos rate   : {y_train.mean():.4f}\n")
    f.write(f"Test  pos rate   : {y_test.mean():.4f}\n")
    f.write(f"Non-zero coefs   : {int((coefs.abs()>1e-10).sum())}/{len(coefs)}\n\n")
    f.write(results.to_string(index=False))
    f.write("\n\nConfusion matrix (L1 Logistic):\n")
    f.write(np.array2string(confusion_matrix(y_test, y_pred)))
print(f"Done. Outputs written to {OUT_DIR}")
