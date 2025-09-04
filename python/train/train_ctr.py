# 1) Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# 2) Load dataset
df = pd.read_csv("python/dataset/online_shoppers_intention.csv")

behavioral = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues"
]
y = df["Revenue"].astype(int)

# 3) Feature engineering (time-based signals)
df["TotalTime"] = (
    df["Administrative_Duration"].clip(lower=0) +
    df["Informational_Duration"].clip(lower=0) +
    df["ProductRelated_Duration"].clip(lower=0)
)
df["Adm_time_per_page"] = df["Administrative_Duration"].clip(lower=0) / (df["Administrative"].clip(lower=0) + 1)
df["Pr_time_per_page"]  = df["ProductRelated_Duration"].clip(lower=0)   / (df["ProductRelated"].clip(lower=0) + 1)

engineered = ["TotalTime", "Adm_time_per_page", "Pr_time_per_page"]
features = behavioral + engineered
X = df[features].copy()

# Basic cleaning
for col in ["BounceRates", "ExitRates"]:
    if col in X.columns:
        X[col] = X[col].clip(0, 1)
X = X.fillna(X.median(numeric_only=True))

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5) Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 6) SMOTE (train only)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_s, y_train)

# 6.5) Hyperparameter tuning (RandomizedSearchCV)
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [200, 500, 800],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 1, 5],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [0, 1, 5]
}

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",   # or "roc_auc"
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train_sm, y_train_sm)
print("Best parameters:", search.best_params_)
print("Best CV score:", search.best_score_)

# 7) Train final XGBoost with best params
best_model = search.best_estimator_
best_model.fit(X_train_sm, y_train_sm)

# 8) Evaluate performance
y_pred  = best_model.predict(X_test_s)
y_proba = best_model.predict_proba(X_test_s)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
mae  = mean_absolute_error(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)

print("\n📊 XGBoost + SMOTE (Behavioral + Engineered)")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"MAE:       {mae:.4f}")
print("Confusion Matrix:\n", cm)

# ========= Complexity via permutation importance + polynomial fits (no SHAP) =========

# Use a manageable sample for speed
rng = np.random.default_rng(42)
idx = rng.choice(X_test_s.shape[0], size=min(2000, X_test_s.shape[0]), replace=False)
X_ref = pd.DataFrame(X_test_s[idx], columns=features)
f = best_model.predict_proba(X_ref.values)[:, 1]  # model probabilities on the sample

# --- NF via permutation importance (AUC drop)
perm = permutation_importance(
    best_model, X_ref, y_test.iloc[idx], scoring="roc_auc",
    n_repeats=10, random_state=42, n_jobs=-1
)
imp = perm.importances_mean
thr = max(1e-4, 0.01 * np.max(imp))  # keep features contributing ≥1% of max importance
NF = int(np.sum(imp > thr))
used_feats = list(np.array(features)[imp > thr])
print(f"\nComplexity — NF: {NF} | Used features: {used_feats}")

# --- MEC: nonlinearity gain per feature (degree-3 vs linear)
MEC = {}
for j, name in enumerate(features):
    xj = X_ref[[name]].values
    # linear fit
    r2_lin = LinearRegression().fit(xj, f).score(xj, f)
    # degree-3 polynomial fit
    Phi3 = PolynomialFeatures(degree=3, include_bias=True).fit_transform(xj)
    r2_poly = LinearRegression().fit(Phi3, f).score(Phi3, f)
    MEC[name] = max(0.0, r2_poly - r2_lin)  # nonlinearity gain


# --- IAS: interaction gain for selected pairs (degree-2 with vs without cross term)
def ias_pair(i_name, j_name):
    xi = X_ref[i_name].values.reshape(-1, 1)
    xj = X_ref[j_name].values.reshape(-1, 1)
    Phi_add  = np.c_[np.ones(len(xi)), xi, xi**2, xj, xj**2]
    r2_add   = LinearRegression().fit(Phi_add, f).score(Phi_add, f)
    Phi_full = np.c_[Phi_add, xi * xj]
    r2_full  = LinearRegression().fit(Phi_full, f).score(Phi_full, f)
    return max(0.0, r2_full - r2_add)

pairs = [("BounceRates","ExitRates"),
         ("ProductRelated","ProductRelated_Duration"),
         ("TotalTime","PageValues")]

# compute & print IAS for selected pairs
IAS_pairs = {f"{a}×{b}": ias_pair(a, b) for a, b in pairs}

# --- Global IAS (all features, degree-2)
Xmat = X_ref.values
Phi_add_all = [np.ones((len(Xmat), 1))]
Phi_add_all += [Xmat[:, [k]] for k in range(Xmat.shape[1])]
Phi_add_all += [Xmat[:, [k]]**2 for k in range(Xmat.shape[1])]
Phi_add_all = np.hstack(Phi_add_all)
r2_add_all = LinearRegression().fit(Phi_add_all, f).score(Phi_add_all, f)

cross_terms = [(Xmat[:, [i]] * Xmat[:, [j]]) for i in range(Xmat.shape[1]) for j in range(i+1, Xmat.shape[1])]
Phi_full_all = np.hstack([Phi_add_all] + cross_terms)
r2_full_all = LinearRegression().fit(Phi_full_all, f).score(Phi_full_all, f)
global_IAS = max(0.0, r2_full_all - r2_add_all)


# # (Optional) save tables
# pd.DataFrame({
#     "feature": list(MEC.keys()),
#     "MEC_nonlin_gain": list(MEC.values()),
#     "perm_importance_auc_drop": imp
# }).to_csv("python/result_ctr/complexity_table_xgb.csv", index=False)
# print("Saved: python/result_ctr/complexity_table_xgb.csv")

# ---- Build 'complexity' dict (so table-row code has what it needs)
# also clean numpy string types to plain str
used_feats = [str(x) for x in used_feats]
complexity = {
    "NF": NF,
    "Used_Features": used_feats,
    "MEC": MEC,
    "IAS_pairs": IAS_pairs,
    "Global_IAS": global_IAS
}

# === Make a single-row summary (like the paper's table) ===
def table_row_for_model(model, model_name, mae, complexity, round_to=2):
    p = model.get_params()
    md = p.get("max_depth", None)
    nr = p.get("n_estimators", None)
    label = f"{model_name} (maxdepth:{md}, nrounds:{nr})"
    mec_total = float(np.nansum(list(complexity["MEC"].values())))  # or np.nanmean(...)
    ias_global = float(complexity["Global_IAS"])
    nf = int(complexity["NF"])
    return pd.DataFrame([{
        "model": label,
        "MAE": round(mae, round_to),
        "MEC": round(mec_total, round_to),
        "IAS": round(ias_global, round_to),
        "NF": nf
    }])

xgb_row = table_row_for_model(best_model, "gbt", mae, complexity, round_to=2)
print("\nSingle-model table row:")
print(xgb_row)
# xgb_row.to_csv("python/result_ctr/table_row_xgb.csv", index=False)
# print("Saved: python/result_ctr/table_row_xgb.csv")
