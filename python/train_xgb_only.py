# train_xgb_only.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def save_cm(y_true, y_pred, classes: List, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(max(8, 0.5 * len(classes)), max(6, 0.5 * len(classes))))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def print_and_save_report(y_true, y_pred, path: str):
    rep = classification_report(y_true, y_pred, digits=5, zero_division=0)
    print(f"\n--- {path} ---\n{rep}")
    with open(path, "w", encoding="utf-8") as f: f.write(rep)

# --- Load preprocessed BASE data ---
train_df = pd.read_csv("train_preprocessed.csv")
cls_df   = pd.read_csv("classified_preprocessed.csv")

features = [c for c in train_df.columns if c != "Class"]
X_full = train_df[features]
y_full = train_df["Class"]
CLASSES = np.unique(y_full)

# Split (stratified)
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

X_train = Xtr_scl
X_val   = Xval_scl
selected = features

# Model: XGBoost only (simple, tweakable)
xgb = XGBClassifier(
    tree_method="hist",
    device="cuda",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    eval_metric="mlogloss",
    random_state=RANDOM_SEED,
)
xgb.fit(X_train, y_train)

# Validation
y_val_pred = xgb.predict(X_val)
acc  = accuracy_score(y_val, y_val_pred)
wf1  = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
mf1  = f1_score(y_val, y_val_pred, average="macro",    zero_division=0)
print(f"\nValidation -> Acc: {acc:.5f} | F1_w: {wf1:.5f} | F1_m: {mf1:.5f}")
print_and_save_report(y_val, y_val_pred, "val_report_xgb_only_base.txt")
save_cm(y_val, y_val_pred, CLASSES, "Confusion Matrix - Validation (XGB only, BASE)", "cm_val_xgb_only_base.png")

# Classified set (same transformers)
Xc_imp = imp.transform(cls_df[features])
Xc_scl = scl.transform(Xc_imp)
Xc_sel = Xc_scl

yc_true = cls_df["Class"]
yc_pred = xgb.predict(Xc_sel)
acc_c  = accuracy_score(yc_true, yc_pred)
wf1_c  = f1_score(yc_true, yc_pred, average="weighted", zero_division=0)
mf1_c  = f1_score(yc_true, yc_pred, average="macro",    zero_division=0)
print(f"Classified -> Acc: {acc_c:.5f} | F1_w: {wf1_c:.5f} | F1_m: {mf1_c:.5f}")
print_and_save_report(yc_true, yc_pred, "classified_report_xgb_only_base.txt")
save_cm(yc_true, yc_pred, CLASSES, "Confusion Matrix - Classified (XGB only, BASE)", "cm_classified_xgb_only_base.png")

# Simple summary CSV (for slides)
pd.DataFrame([{
    "Setting":"XGB_only_BASE",
    "Val_Accuracy":acc, "Val_F1_weighted":wf1, "Val_F1_macro":mf1,
    "Cls_Accuracy":acc_c,"Cls_F1_weighted":wf1_c,"Cls_F1_macro":mf1_c
}]).to_csv("summary_xgb_only_base.csv", index=False)

print("\nArtifacts saved:")
for fn in [
    "val_report_xgb_only_base.txt", "cm_val_xgb_only_base.png",
    "classified_report_xgb_only_base.txt", "cm_classified_xgb_only_base.png",
    "summary_xgb_only_base.csv"
]:
    print(" -", fn)

