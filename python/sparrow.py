import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import boxcox

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
classified_df = pd.read_csv("classified.csv")

# Drop rows with unknown class (-1) from classified set
classified_df = classified_df[classified_df["Class"] != -1]

# Drop unnecessary columns
drop_cols = ["Artist Name", "Track Name", "Popularity"]
train_df_cleaned = train_df.drop(columns=drop_cols)
classified_df_cleaned = classified_df.drop(columns=drop_cols)
test_df_cleaned = test_df.copy()

# Fix duration values < 30
duration_col = "duration_in min/ms"
for df in [train_df_cleaned, test_df_cleaned, classified_df_cleaned]:
    condition = df[duration_col] < 30
    df.loc[condition, duration_col] *= 60000

# Fill missing instrumentalness with training median
instr_mean = train_df_cleaned["instrumentalness"].mean()
for df in [train_df_cleaned, test_df_cleaned, classified_df_cleaned]:
    df["instrumentalness"] = df["instrumentalness"].fillna(instr_mean)

# Fill missing 'key' with -1
for df in [train_df_cleaned, test_df_cleaned, classified_df_cleaned]:
    df["key"] = df["key"].fillna(-1)

# Save outputs
train_df_cleaned.to_csv("train_preprocessed.csv", index=False)
test_df_cleaned.to_csv("test_preprocessed.csv", index=False)
classified_df_cleaned.to_csv("classified_preprocessed.csv", index=False)




# === Load datasets ===
train_df = pd.read_csv("train_preprocessed.csv")
classified_df = pd.read_csv("classified_preprocessed.csv")

# === Prepare features and targets ===
features = [col for col in train_df.columns if col not in ["Class", "Artist Name"]]
X = train_df[features]
y = train_df["Class"]

# === Impute and scale ===
imp = SimpleImputer(strategy="mean")
scl = StandardScaler()
X_imp = imp.fit_transform(X)
X_scl = scl.fit_transform(X_imp)

# === SelectKBest (top 15 features) ===
selector = SelectKBest(score_func=f_classif, k=15)
X_sel = selector.fit_transform(X_scl, y)
selected_features = np.array(features)[selector.get_support()]
print("Selected features (SelectKBest):", selected_features.tolist())

# === Stratified split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_sel, y, test_size=0.2, stratify=y, random_state=42
)

# === Base models ===
rf = RandomForestClassifier(n_estimators=500, random_state=42)
xgb = XGBClassifier(tree_method="hist",device="cuda", eval_metric="mlogloss", n_estimators=100, max_depth=6)

# === Meta model ===
meta = LogisticRegression(max_iter=1000)

# === Stacking classifier ===
stack = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    final_estimator=meta,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

# === Train model ===
stack.fit(X_train, y_train)

# === Evaluate on validation set ===
y_pred = stack.predict(X_val)
acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)

print("\\n=== Validation Results ===")
print(f"Accuracy : {acc:.5f}")
print(f"Precision: {prec:.5f}")
print(f"Recall   : {rec:.5f}")
print(f"F1 Score : {f1:.5f}")

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Validation")
plt.tight_layout()
plt.savefig("confusion_matrix_val_selected.png")

# === Evaluate on classified set ===
Xc = classified_df[selected_features]
yc = classified_df["Class"]
Xc_imp = imp.transform(Xc)
Xc_scl = scl.transform(Xc_imp)
Xc_sel = selector.transform(Xc_scl)

yc_pred = stack.predict(Xc_sel)
acc_c = accuracy_score(yc, yc_pred)
prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(yc, yc_pred, average="weighted", zero_division=0)

print("\\n=== Classified Results ===")
print(f"Accuracy : {acc_c:.5f}")
print(f"Precision: {prec_c:.5f}")
print(f"Recall   : {rec_c:.5f}")
print(f"F1 Score : {f1_c:.5f}")

cm_c = confusion_matrix(yc, yc_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_c, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Classified")
plt.tight_layout()
plt.savefig("confusion_matrix_classified_selected.png")