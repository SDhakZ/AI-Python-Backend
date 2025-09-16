# 1) Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# For mRMR
from mrmr import mrmr_classif   # pip install mrmr_selection

# 2) Load preprocessed dataset
df = pd.read_csv("python/dataset/processed_online_shoppers.csv")

# Features (all behavioral already preprocessed)
behavioral = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues"
]
X = df[behavioral].copy()
y = df["Revenue"].astype(int)

# 3) Feature selection with mRMR
selected_feats = mrmr_classif(X=X, y=y, K=min(15, X.shape[1]))
print("Selected features by mRMR:", selected_feats)
X = X[selected_feats]

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5) SMOTE (only on training set)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 6) Base XGBoost (no tuning)
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
    max_depth=6,       
    n_estimators=100   
)

xgb_clf.fit(X_train_sm, y_train_sm)

# 7) Evaluate performance
y_pred  = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
mae  = mean_absolute_error(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)

print("\n📊 Base XGBoost + SMOTE + mRMR")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"MAE:       {mae:.4f}")
print("Confusion Matrix:\n", cm)

# ========= Complexity Analysis =========

rng = np.random.default_rng(42)
idx = rng.choice(X_test.shape[0], size=min(2000, X_test.shape[0]), replace=False)
X_ref = pd.DataFrame(X_test.iloc[idx], columns=selected_feats)
f = xgb_clf.predict_proba(X_ref.values)[:, 1]

# NF (number of used features via permutation importance)
perm = permutation_importance(
    xgb_clf, X_ref, y_test.iloc[idx], scoring="roc_auc",
    n_repeats=10, random_state=42, n_jobs=1
)
imp = perm.importances_mean
thr = max(1e-4, 0.01 * np.max(imp))
NF = int(np.sum(imp > thr))

# MEC (main effect complexity)
MEC = {}
for j, name in enumerate(selected_feats):
    xj = X_ref[[name]].values
    r2_lin = LinearRegression().fit(xj, f).score(xj, f)
    Phi3 = PolynomialFeatures(degree=3, include_bias=True).fit_transform(xj)
    r2_poly = LinearRegression().fit(Phi3, f).score(Phi3, f)
    MEC[name] = max(0.0, r2_poly - r2_lin)
mec_total = float(np.nansum(list(MEC.values())))

# Global IAS (interaction strength)
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

# Table row
xgb_row = pd.DataFrame([{
    "model": f"gbt (maxdepth:{xgb_clf.get_params().get('max_depth')}, nrounds:{xgb_clf.get_params().get('n_estimators')})",
    "MAE": round(mae, 2),
    "MEC": round(mec_total, 2),
    "IAS": round(global_IAS, 2),
    "NF": NF
}])

print("\nSingle-model table row:")
print(xgb_row)
