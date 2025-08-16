# train_ensemble_popkey_meta_xgb.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def save_cm(y_true, y_pred, classes: List, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(max(8, 0.5 * len(classes)), max(6, 0.5 * len(classes)) ))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def print_and_save_report(y_true, y_pred, path: str):
    rep = classification_report(y_true, y_pred, digits=5, zero_division=0)
    print(f"\n--- {path} ---\n{rep}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep)

# --- Load preprocessed POP+KEY data ---
train_df = pd.read_csv("train_preprocessed_popkey.csv")
cls_df   = pd.read_csv("classified_preprocessed_popkey.csv")

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
X_train = scl.transform(Xtr_imp)
X_val   = scl.transform(Xval_imp)

# --- Base learners ---
rf = RandomForestClassifier(
    n_estimators=800,
    random_state=RANDOM_SEED
)

xgb_base = XGBClassifier(
    tree_method="hist",
    n_estimators=900,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=RANDOM_SEED,
)

# --- Meta learner: XGBoost (regularized & shallow to reduce overfit) ---
xgb_meta = XGBClassifier(
    tree_method="hist",
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=RANDOM_SEED,
)

# Stacking (passthrough so meta sees OOF preds + original features)
stack = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb_base)],
    final_estimator=xgb_meta,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

# Train
stack.fit(X_train, y_train)

# Validation
y_val_pred = stack.predict(X_val)
acc  = accuracy_score(y_val, y_val_pred)
wf1  = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
mf1  = f1_score(y_val, y_val_pred, average="macro",    zero_division=0)
print(f"\nValidation -> Acc: {acc:.5f} | F1_w: {wf1:.5f} | F1_m: {mf1:.5f}")
print_and_save_report(y_val, y_val_pred, "val_report_ensemble_popkey_metaXGB.txt")
save_cm(y_val, y_val_pred, CLASSES,
        "Confusion Matrix - Validation (Ensemble POP+KEY, meta=XGB)",
        "cm_val_ensemble_popkey_metaXGB.png")

# Classified set (same transformers)
Xc = scl.transform(imp.transform(cls_df[features]))
yc_true = cls_df["Class"]
yc_pred = stack.predict(Xc)
acc_c  = accuracy_score(yc_true, yc_pred)
wf1_c  = f1_score(yc_true, yc_pred, average="weighted", zero_division=0)
mf1_c  = f1_score(yc_true, yc_pred, average="macro",    zero_division=0)
print(f"Classified -> Acc: {acc_c:.5f} | F1_w: {wf1_c:.5f} | F1_m: {mf1_c:.5f}")
print_and_save_report(yc_true, yc_pred, "classified_report_ensemble_popkey.txt")
save_cm(yc_true, yc_pred, CLASSES, "Confusion Matrix - Classified (Ensemble POP+KEY)", "cm_classified_ensemble_popkey.png")

# Simple summary CSV (for slides)
pd.DataFrame([{
    "Setting":"Ensemble_POP+KEY",
    "Val_Accuracy":acc, "Val_F1_weighted":wf1, "Val_F1_macro":mf1,
    "Cls_Accuracy":acc_c,"Cls_F1_weighted":wf1_c,"Cls_F1_macro":mf1_c
}]).to_csv("summary_ensemble_popkey.csv", index=False)

print("\nArtifacts saved:")
for fn in [
    "val_report_ensemble_popkey.txt", "cm_val_ensemble_popkey.png",
    "classified_report_ensemble_popkey.txt", "cm_classified_ensemble_popkey.png",
    "summary_ensemble_popkey.csv"
]:
    print(" -", fn)

# ===================== Feature Importance Plots =====================
# 1) Base XGB feature importances (on original features)
try:
    base_xgb = stack.named_estimators_["xgb"]
    importances_base = base_xgb.feature_importances_
    idx_base = np.argsort(importances_base)

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(features)[idx_base], importances_base[idx_base])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance — Base XGB in Ensemble (original features)")
    plt.tight_layout()
    plt.savefig("feature_importance_base_xgb.png", dpi=150)
    plt.close()
    print("Saved: feature_importance_base_xgb.png")
except Exception as e:
    print("Base XGB importance plot skipped:", e)

# 2) Meta XGB feature importances (on OOF preds + original features)
try:
    meta = stack.final_estimator_
    importances_meta = meta.feature_importances_

    # Figure out how many meta-features there are
    n_meta_in = getattr(meta, "n_features_in_", None)
    if n_meta_in is None:
        n_meta_in = importances_meta.shape[0]

    # Passthrough adds original features at the end
    passthrough = getattr(stack, "passthrough", False)
    n_orig_feats = len(features) if passthrough else 0

    # Number of columns coming from base estimators' outputs
    n_pred_cols = n_meta_in - n_orig_feats
    n_estimators = len(stack.estimators)
    # Columns per estimator (usually equals number of classes if using predict_proba)
    per_estimator_cols = n_pred_cols // n_estimators if n_estimators > 0 else 0

    # Build names for base predictions; prefer class-aware names if sizes match
    # CLASSES is defined earlier in your script
    base_pred_names = []
    for name, _ in stack.estimators:
        if per_estimator_cols == len(CLASSES):
            base_pred_names += [f"{name}_proba_{cls}" for cls in CLASSES]
        else:
            base_pred_names += [f"{name}_feat_{i}" for i in range(per_estimator_cols)]

    meta_feature_names = base_pred_names + (list(features) if passthrough else [])

    # Safety: align lengths in case of off-by-ones on some setups
    L = min(len(importances_meta), len(meta_feature_names))
    importances_meta = importances_meta[:L]
    meta_feature_names = meta_feature_names[:L]

    idx_meta = np.argsort(importances_meta)
    plt.figure(figsize=(12, 8))
    plt.barh(np.array(meta_feature_names)[idx_meta], importances_meta[idx_meta])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance — Meta XGB (OOF preds + original features)")
    plt.tight_layout()
    plt.savefig("feature_importance_meta_xgb.png", dpi=150)
    plt.close()
    print("Saved: feature_importance_meta_xgb.png")
except Exception as e:
    print("Meta XGB importance plot skipped:", e)
# ====================================================================
