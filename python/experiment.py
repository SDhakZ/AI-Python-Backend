# run_popularity_and_cyclic_key_experiments.py
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------
# Utilities
# ---------------------------
def save_cm(y_true, y_pred, classes: List, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(max(8, 0.5 * len(classes)), max(6, 0.5 * len(classes))))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def print_and_save_report(y_true, y_pred, path: str):
    rep = classification_report(y_true, y_pred, digits=5, zero_division=0)
    print(f"\n--- {path} ---\n{rep}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep)

def cyclic_encode_key(key_series: pd.Series, n_cycle: int = 12) -> pd.DataFrame:
    """
    Expects musical key integers in [0..11]. Non-valid (<0 or NaN) -> NaN.
    Returns DataFrame with columns key_sin, key_cos.
    """
    k = key_series.copy()
    k = k.where((k >= 0) & (k < n_cycle))  # invalid -> NaN
    angle = 2 * np.pi * (k / n_cycle)
    return pd.DataFrame({
        "key_sin": np.sin(angle),
        "key_cos": np.cos(angle),
    })

def common_preclean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common cleanups shared by both experiments:
    - fix duration units (values < 30 are minutes -> convert to ms)
    - fill instrumentalness with training mean later (we'll compute using train only)
    """
    out = df.copy()
    # Convert duration if < 30 -> assume minutes
    dur_col = "duration_in min/ms"
    if dur_col in out.columns:
        mask = out[dur_col] < 30
        out.loc[mask, dur_col] = out.loc[mask, dur_col] * 60000
    return out

# ---------------------------
# 1) Create two preprocessed datasets
# ---------------------------
train_raw = pd.read_csv("train.csv")
classified_raw = pd.read_csv("classified.csv")
classified_raw = classified_raw[classified_raw["Class"] != -1]  # drop unknowns

# --- EXP 1: Base (drop Popularity, keep key as-is with fill -1)
drop_cols_base = ["Artist Name", "Track Name", "Popularity"]
train_base = common_preclean(train_raw.drop(columns=drop_cols_base))
cls_base   = common_preclean(classified_raw.drop(columns=drop_cols_base))

# Fill instrumentalness with TRAIN mean (use base train to compute)
instr_mean_base = train_base["instrumentalness"].mean()
for df in (train_base, cls_base):
    df["instrumentalness"] = df["instrumentalness"].fillna(instr_mean_base)
    df["key"] = df["key"].fillna(-1)  # plain key with -1 for missing

train_base.to_csv("train_preprocessed.csv", index=False)
cls_base.to_csv("classified_preprocessed.csv", index=False)

# --- EXP 2: Pop+Key (keep Popularity, cyclic-encode key, drop original key)
drop_cols_popkey = ["Artist Name", "Track Name"]  # keep Popularity this time
train_pk = common_preclean(train_raw.drop(columns=drop_cols_popkey))
cls_pk   = common_preclean(classified_raw.drop(columns=drop_cols_popkey))

# Fill instrumentalness with TRAIN mean (use pk train to compute)
instr_mean_pk = train_pk["instrumentalness"].mean()
for df in (train_pk, cls_pk):
    df["instrumentalness"] = df["instrumentalness"].fillna(instr_mean_pk)

# Cyclic encode key -> add key_sin, key_cos; then drop original 'key'
train_key_enc = cyclic_encode_key(train_pk["key"])
cls_key_enc   = cyclic_encode_key(cls_pk["key"])

train_pk = pd.concat([train_pk.drop(columns=["key"]), train_key_enc], axis=1)
cls_pk   = pd.concat([cls_pk.drop(columns=["key"]), cls_key_enc], axis=1)

train_pk.to_csv("train_preprocessed_popkey.csv", index=False)
cls_pk.to_csv("classified_preprocessed_popkey.csv", index=False)

print("Saved preprocessed datasets:")
print(" - train_preprocessed.csv / classified_preprocessed.csv      (BASE)")
print(" - train_preprocessed_popkey.csv / classified_preprocessed_popkey.csv (POP+KEY)")

# ---------------------------
# 2) Modeling helper (leakage-safe)
# ---------------------------
def run_experiment(train_path: str, cls_path: str, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/evaluate the stacking model on a given preprocessed dataset.
    Saves reports/plots with filename suffix f"_{tag}".
    Returns (validation_summary_row_df, classified_summary_row_df)
    """
    train_df = pd.read_csv(train_path)
    cls_df   = pd.read_csv(cls_path)

    # Features and target
    features = [c for c in train_df.columns if c != "Class"]
    X_full = train_df[features]
    y_full = train_df["Class"]
    CLASSES = np.unique(y_full)

    # Split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.20, stratify=y_full, random_state=RANDOM_SEED
    )

    # Leakage-safe preprocessing
    imp = SimpleImputer(strategy="mean").fit(X_train_raw)
    Xtr_imp  = imp.transform(X_train_raw)
    Xval_imp = imp.transform(X_val_raw)

    scl = StandardScaler().fit(Xtr_imp)
    Xtr_scl  = scl.transform(Xtr_imp)
    Xval_scl = scl.transform(Xval_imp)

    k = min(15, Xtr_scl.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k).fit(Xtr_scl, y_train)
    X_train = selector.transform(Xtr_scl)
    X_val   = selector.transform(Xval_scl)

    selected_feature_names = np.array(features)[selector.get_support()]
    print(f"\n[{tag}] Selected features ({len(selected_feature_names)}):", selected_feature_names.tolist())

    # Models
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED)
    xgb = XGBClassifier(
        tree_method="hist",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=meta,
        passthrough=True,
        cv=5,
        n_jobs=-1
    )

    # Train & validate
    stack.fit(X_train, y_train)
    y_val_pred = stack.predict(X_val)

    acc  = accuracy_score(y_val, y_val_pred)
    wf1  = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    mf1  = f1_score(y_val, y_val_pred, average="macro",    zero_division=0)
    print(f"\n[{tag}] Validation -> Acc: {acc:.5f} | F1_w: {wf1:.5f} | F1_m: {mf1:.5f}")

    print_and_save_report(y_val, y_val_pred, f"val_report_{tag}.txt")
    save_cm(y_val, y_val_pred, CLASSES,
            f"Confusion Matrix - Validation ({tag})",
            f"cm_val_{tag}.png")

    # Evaluate on classified set (transform with the same train-fitted steps)
    Xc_full = cls_df[features]
    yc_full = cls_df["Class"]
    Xc_imp  = imp.transform(Xc_full)
    Xc_scl  = scl.transform(Xc_imp)
    Xc_sel  = selector.transform(Xc_scl)

    yc_pred = stack.predict(Xc_sel)

    acc_c = accuracy_score(yc_full, yc_pred)
    wf1_c = f1_score(yc_full, yc_pred, average="weighted", zero_division=0)
    mf1_c = f1_score(yc_full, yc_pred, average="macro",    zero_division=0)
    print(f"[{tag}] Classified -> Acc: {acc_c:.5f} | F1_w: {wf1_c:.5f} | F1_m: {mf1_c:.5f}")

    print_and_save_report(yc_full, yc_pred, f"classified_report_{tag}.txt")
    save_cm(yc_full, yc_pred, CLASSES,
            f"Confusion Matrix - Classified ({tag})",
            f"cm_classified_{tag}.png")

    # Summaries for slide tables
    val_row = pd.DataFrame([{
        "Setting": tag, "Accuracy": acc, "F1_weighted": wf1, "F1_macro": mf1
    }])
    cls_row = pd.DataFrame([{
        "Setting": tag, "Accuracy": acc_c, "F1_weighted": wf1_c, "F1_macro": mf1_c
    }])

    return val_row, cls_row

# ---------------------------
# 3) Run both experiments
# ---------------------------
val_rows = []
cls_rows = []

# Experiment 1: BASE (no Popularity, raw key with -1)
vr1, cr1 = run_experiment("train_preprocessed.csv", "classified_preprocessed.csv", tag="base_no_pop")
val_rows.append(vr1)
cls_rows.append(cr1)

# Experiment 2: POP+KEY (keep Popularity, cyclic key encoding)
vr2, cr2 = run_experiment("train_preprocessed_popkey.csv", "classified_preprocessed_popkey.csv", tag="with_pop_cyclic_key")
val_rows.append(vr2)
cls_rows.append(cr2)

summary_val = pd.concat(val_rows, ignore_index=True)
summary_cls = pd.concat(cls_rows, ignore_index=True)

print("\n=== Validation Summary (both experiments) ===\n", summary_val.round(5))
print("\n=== Classified Summary (both experiments) ===\n", summary_cls.round(5))

summary_val.to_csv("experiments_validation_summary.csv", index=False)
summary_cls.to_csv("experiments_classified_summary.csv", index=False)

print("\nArtifacts saved:")
for fn in [
    # datasets
    "train_preprocessed.csv", "classified_preprocessed.csv",
    "train_preprocessed_popkey.csv", "classified_preprocessed_popkey.csv",
    # summaries
    "experiments_validation_summary.csv", "experiments_classified_summary.csv",
    # reports & matrices (per experiment)
    "val_report_base_no_pop.txt", "cm_val_base_no_pop.png",
    "classified_report_base_no_pop.txt", "cm_classified_base_no_pop.png",
    "val_report_with_pop_cyclic_key.txt", "cm_val_with_pop_cyclic_key.png",
    "classified_report_with_pop_cyclic_key.txt", "cm_classified_with_pop_cyclic_key.png",
]:
    print(" -", fn)   
