import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from tqdm import tqdm

# === 0) Load preprocessed data ===
train_df = pd.read_csv("train_preprocessed.csv")

# === 1) Build super‑genre labels ===
GROUP_MAP = {
    0: 0, 4: 0,               # Acoustic/Folk & Country
    1: 1, 6: 1, 9: 1,         # Alt/Indie/Pop
    2: 2, 8: 2, 10: 2,        # Rock/Metal/Blues
    5: 3, 7: 3,               # HipHop/Instrumental
    3: 4                      # Bollywood
}
train_df["Group"] = train_df["Class"].map(GROUP_MAP)

# === 2) Set feature columns (exclude non-numeric + target columns) ===
features = [col for col in train_df.columns if col not in ["Class", "Group", "Artist Name"]]

# === 3) Setup cross-validation ===
kf = KFold(n_splits=4, shuffle=True, random_state=42)
metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"\n📦 Fold {fold+1}/4")

    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_val   = train_df.iloc[val_idx].reset_index(drop=True)

    # --- Stage 1: train super-genre classifier ---
    imp1 = SimpleImputer(strategy="mean")
    scl1 = StandardScaler()
    X1_imp = imp1.fit_transform(df_train[features])
    X1_scl = scl1.fit_transform(X1_imp)
    y1 = df_train["Group"].values

    dtrain1 = xgb.QuantileDMatrix(X1_scl, y1)
    params1 = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "eta": 0.2,
        "verbosity": 0,
        "num_class": len(np.unique(y1))
    }

    bst1 = xgb.train(
        params1,
        dtrain1,
        num_boost_round=1000,
        evals=[(dtrain1, "train")],
        verbose_eval=False
    )

    # --- Stage 2: train sub-genre classifiers per group ---
    sub_models = {}
    for gid in sorted(df_train["Group"].unique()):
        sub = df_train[df_train["Group"] == gid]
        y_sub = sub["Class"].values
        unique_sub = np.unique(y_sub)

        if len(unique_sub) == 1:
            sub_models[gid] = {"const": unique_sub[0]}
        else:
            X_sub = sub[features]
            le = LabelEncoder()
            y_enc = le.fit_transform(y_sub)

            imp2 = SimpleImputer(strategy="mean")
            scl2 = StandardScaler()
            X2_imp = imp2.fit_transform(X_sub)
            X2_scl = scl2.fit_transform(X2_imp)

            dtrain2 = xgb.QuantileDMatrix(X2_scl, y_enc)
            params2 = params1.copy()
            params2["num_class"] = len(le.classes_)

            bst2 = xgb.train(
                params2,
                dtrain2,
                num_boost_round=1200,
                evals=[(dtrain2, "train")],
                verbose_eval=False
            )

            sub_models[gid] = {
                "imp": imp2,
                "scl": scl2,
                "bst": bst2,
                "le": le
            }

    # --- Evaluation on validation set ---
    X_val = df_val[features]
    y_true = df_val["Class"].values
    X1v = scl1.transform(imp1.transform(X_val))
    grp_pred = np.argmax(bst1.predict(xgb.QuantileDMatrix(X1v)), axis=1)

    preds = np.zeros(len(X_val), dtype=int)
    for gid, info in sub_models.items():
        idx = np.where(grp_pred == gid)[0]
        if idx.size:
            if "const" in info:
                preds[idx] = info["const"]
            else:
                Xe = X_val.iloc[idx]
                X2e_imp = info["imp"].transform(Xe)
                X2e_scl = info["scl"].transform(X2e_imp)
                p2 = info["bst"].predict(xgb.QuantileDMatrix(X2e_scl))
                sub_enc = np.argmax(p2, axis=1)
                preds[idx] = info["le"].inverse_transform(sub_enc)

    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1‑Score:  {f1:.4f}")

    metrics.append((acc, prec, rec, f1))

# === Aggregate results ===
accs, precs, recs, f1s = zip(*metrics)
print("\n📊 Cross-Validation Summary:")
print(f"  Mean Accuracy:  {np.mean(accs):.4f}")
print(f"  Mean Precision: {np.mean(precs):.4f}")
print(f"  Mean Recall:    {np.mean(recs):.4f}")
print(f"  Mean F1‑Score:  {np.mean(f1s):.4f}")
