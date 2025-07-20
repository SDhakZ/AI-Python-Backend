import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# === 0) Load preprocessed data ===
train_df = pd.read_csv("train_preprocessed.csv")
test_df  = pd.read_csv("test_preprocessed.csv")

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

# === 3) Stage 1: Train super-genre classifier ===
print("▶ Stage 1: training super‑genre model")
imp1 = SimpleImputer(strategy="mean")
scl1 = StandardScaler()

X1_imp = imp1.fit_transform(train_df[features])
X1_scl = scl1.fit_transform(X1_imp)
y1 = train_df["Group"].values

dtrain1 = xgb.QuantileDMatrix(X1_scl, y1)
params1 = {
    "device": "cuda",
    "tree_method": "hist",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "eta": 0.2,
    "verbosity": 1,
    "num_class": len(np.unique(y1))
}

bst1 = xgb.train(
    params1,
    dtrain1,
    num_boost_round=1000,
    evals=[(dtrain1, "train")],
    verbose_eval=10
)
print("✔ Stage 1 done.\n")

# === 4) Stage 2: Train sub-genre classifiers ===
print("▶ Stage 2: training sub‑genre models")
sub_models = {}

for gid in tqdm(sorted(train_df["Group"].unique()), desc="Groups"):
    sub = train_df[train_df["Group"] == gid]
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
            verbose_eval=10
        )

        sub_models[gid] = {
            "imp": imp2,
            "scl": scl2,
            "bst": bst2,
            "le": le
        }

print("✔ Stage 2 done.\n")

# === 5) Evaluate on train set ===
print("▶ Evaluating on train set")
X_eval = train_df[features]
y_true = train_df["Class"].values

X1e = scl1.transform(imp1.transform(X_eval))
grp_pred = np.argmax(bst1.predict(xgb.QuantileDMatrix(X1e)), axis=1)

preds = np.zeros(len(X_eval), dtype=int)
for gid, info in sub_models.items():
    idx = np.where(grp_pred == gid)[0]
    if idx.size:
        if "const" in info:
            preds[idx] = info["const"]
        else:
            Xe = X_eval.iloc[idx]
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
print(f"  F1‑Score:  {f1:.4f}\n")

# === 6) Predict on test set & save === 
print("▶ Predicting on test set")
X_test = test_df[features]
X1t = scl1.transform(imp1.transform(X_test))
grp_test = np.argmax(bst1.predict(xgb.QuantileDMatrix(X1t)), axis=1)

final = np.zeros(len(X_test), dtype=int)
for gid, info in sub_models.items():
    idx = np.where(grp_test == gid)[0]
    if idx.size:
        if "const" in info:
            final[idx] = info["const"]
        else:
            Xe = X_test.iloc[idx]
            X2t_imp = info["imp"].transform(Xe)
            X2t_scl = info["scl"].transform(X2t_imp)
            p2 = info["bst"].predict(xgb.QuantileDMatrix(X2t_scl))
            sub_enc = np.argmax(p2, axis=1)
            final[idx] = info["le"].inverse_transform(sub_enc)

test_df["Class"] = final
test_df[["Artist Name", "Track Name", "Class"]].to_csv("test_predictions.csv", index=False)
print("✔ Saved test_predictions.csv")
