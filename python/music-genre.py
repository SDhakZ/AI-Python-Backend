import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# === Load data ===
train_df = pd.read_csv("train.csv")

# === Define genre groups ===
group_map = {
    0: 0, 4: 0,
    1: 1, 6: 1, 9: 1,
    2: 2, 8: 2, 10: 2,
    5: 3, 7: 3,
    3: 4
}
train_df['Group'] = train_df['Class'].map(group_map)

# === Impute missing instrumentalness by genre-wise mean ===
genre_means = train_df.groupby("Class")["instrumentalness"].mean()
train_df["instrumentalness"] = train_df.apply(
    lambda r: genre_means.loc[r["Class"]] if pd.isna(r["instrumentalness"]) else r["instrumentalness"],
    axis=1
)

# === Feature columns (drop metadata + targets) ===
features = [c for c in train_df.columns if c not in ["Artist Name","Track Name","Class","Group"]]

# === Stratified K-Fold Cross-Validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = []
all_truth = []

print("▶ Running Stratified 5-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["Class"]), 1):
    print(f"→ Fold {fold}")
    train_fold = train_df.iloc[train_idx]
    val_fold   = train_df.iloc[val_idx]

    # Stage 1: Train super‑genre classifier
    group_pipeline = Pipeline([
        ("imputer",    SimpleImputer(strategy="mean")),
        ("scaler",     StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=10000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ])
    group_pipeline.fit(train_fold[features], train_fold["Group"])

    # Stage 2: Sub-genre classifiers per group
    group_genre_classifiers = {}
    for gid in sorted(train_fold["Group"].unique()):
        sub = train_fold[train_fold["Group"] == gid]
        X_sub = sub[features]
        y_sub = sub["Class"]
        clf = Pipeline([
            ("imputer",    SimpleImputer(strategy="mean")),
            ("scaler",     StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=10000,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ))
        ])
        clf.fit(X_sub, y_sub)
        group_genre_classifiers[gid] = clf

    # Evaluate on validation fold
    X_val = val_fold[features]
    y_true = val_fold["Class"]
    grp_preds = group_pipeline.predict(X_val)
    preds = np.zeros(len(X_val), dtype=int)

    for gid, clf in group_genre_classifiers.items():
        idx = np.where(grp_preds == gid)[0]
        if idx.size:
            preds[idx] = clf.predict(X_val.iloc[idx])

    all_preds.extend(preds)
    all_truth.extend(y_true)

# === Final Evaluation ===
acc = accuracy_score(all_truth, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_truth, all_preds, average="weighted")

print("\\n=== Cross-Validation Results ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")