import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# === Load data ===
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# === Define genre groups ===
group_map = {
    0: 0, 4: 0,               # Acoustic/Folk & Country
    1: 1, 6: 1, 9: 1,         # Alt/Indie/Pop
    2: 2, 8: 2, 10: 2,        # Rock/Metal/Blues
    5: 3, 7: 3,               # HipHop/Instrumental
    3: 4                      # Bollywood
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

# === Stage 1: Super‑genre classifier ===
print("▶ Training super‑genre classifier...")
group_pipeline = Pipeline([
    ("imputer",    SimpleImputer(strategy="mean")),
    ("scaler",     StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        verbose=1,               # prints tree‑building progress
        random_state=42
    ))
])
group_pipeline.fit(train_df[features], train_df['Group'])
print("✔ Super‑genre model done.\n")

# === Stage 2: One classifier per group ===
group_genre_classifiers = {}
print("▶ Training sub‑genre classifiers:")
for group_id in tqdm(sorted(train_df['Group'].unique()), desc="Groups"):
    sub = train_df[train_df['Group']==group_id]
    X_sub = sub[features]
    y_sub = sub['Class']
    clf = Pipeline([
        ("imputer",    SimpleImputer(strategy="mean")),
        ("scaler",     StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            verbose=1,           # per‑tree output
            random_state=42
        ))
    ])
    clf.fit(X_sub, y_sub)
    group_genre_classifiers[group_id] = clf
print("✔ All sub‑genre models done.\n")

# === Evaluate on training set (since test labels are unknown) ===
print("▶ Evaluating on train set...")
X_eval = train_df[features]
y_true = train_df['Class']

grp_preds = group_pipeline.predict(X_eval)
final_preds = np.zeros(len(X_eval), dtype=int)

for gid, clf in group_genre_classifiers.items():
    idx = np.where(grp_preds == gid)[0]
    if idx.size:
        final_preds[idx] = clf.predict(X_eval.iloc[idx])

acc = accuracy_score(y_true, final_preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, final_preds, average='weighted')
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1‑Score:  {f1:.4f}\n")

# === Predict on test set and save ===
print("▶ Predicting on test set...")
X_test_feat      = test_df[features]
grp_preds_test   = group_pipeline.predict(X_test_feat)
final_test_preds = np.zeros(len(X_test_feat), dtype=int)

for gid, clf in group_genre_classifiers.items():
    idx = np.where(grp_preds_test == gid)[0]
    if idx.size:
        final_test_preds[idx] = clf.predict(X_test_feat.iloc[idx])

test_df['Class'] = final_test_preds
test_df[["Artist Name","Track Name","Class"]].to_csv("test_predictions.csv", index=False)
print("✔ Saved test_predictions.csv")
