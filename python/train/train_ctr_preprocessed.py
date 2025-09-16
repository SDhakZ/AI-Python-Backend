# =========================
# 1) Imports
# =========================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from mrmr import mrmr_classif   # pip install mrmr_selection

# =========================
# 2) Load dataset
# =========================
df = pd.read_csv("python/dataset/processed_online_shoppers.csv")

X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)


# =========================
# 4) Cross-validation setup
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n===== Fold {fold} =====")

    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # SMOTE on training set
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # Model
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",                # remove if no GPU available
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        max_depth=6,
        n_estimators=100
    )
    xgb_clf.fit(X_train_sm, y_train_sm)

    # Predictions
    y_pred = xgb_clf.predict(X_test)
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]

    # Performance metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    mae  = mean_absolute_error(y_test, y_proba)

    # =========================
    # Complexity Analysis
    # =========================
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test.shape[0], size=min(2000, X_test.shape[0]), replace=False)
    X_ref = pd.DataFrame(X_test.iloc[idx], columns=X_test.columns)
    f_hat = xgb_clf.predict_proba(X_ref.values)[:, 1]

    # NF (features used via permutation importance)
    perm = permutation_importance(
        xgb_clf, X_ref, y_test.iloc[idx], scoring="roc_auc",
        n_repeats=10, random_state=42, n_jobs=1
    )
    imp = perm.importances_mean
    thr = max(1e-4, 0.01 * np.max(imp))
    NF = int(np.sum(imp > thr))

    # MEC (main effect complexity)
    MEC = {}
    for j, name in enumerate(X_ref.columns):
        xj = X_ref[[name]].values
        r2_lin = LinearRegression().fit(xj, f_hat).score(xj, f_hat)
        Phi3 = PolynomialFeatures(degree=3, include_bias=True).fit_transform(xj)
        r2_poly = LinearRegression().fit(Phi3, f_hat).score(Phi3, f_hat)
        MEC[name] = max(0.0, r2_poly - r2_lin)
    mec_total = float(np.nansum(list(MEC.values())))

    # IAS (interaction strength)
    Xmat = X_ref.values
    Phi_add_all = [np.ones((len(Xmat), 1))]
    Phi_add_all += [Xmat[:, [k]] for k in range(Xmat.shape[1])]
    Phi_add_all += [Xmat[:, [k]]**2 for k in range(Xmat.shape[1])]
    Phi_add_all = np.hstack(Phi_add_all)
    r2_add_all = LinearRegression().fit(Phi_add_all, f_hat).score(Phi_add_all, f_hat)

    cross_terms = [(Xmat[:, [i]] * Xmat[:, [j]]) 
                   for i in range(Xmat.shape[1]) for j in range(i+1, Xmat.shape[1])]
    Phi_full_all = np.hstack([Phi_add_all] + cross_terms)
    r2_full_all = LinearRegression().fit(Phi_full_all, f_hat).score(Phi_full_all, f_hat)
    IAS = max(0.0, r2_full_all - r2_add_all)

    # Store results
    results.append({
        "fold": fold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "mae": mae,
        "NF": NF,
        "MEC": mec_total,
        "IAS": IAS
    })

# =========================
# 5) Results summary
# =========================
df_results = pd.DataFrame(results)
print("\n=== Fold-wise Results ===")
print(df_results)

print("\n=== Mean Results Across 5 Folds ===")
print(df_results.mean(numeric_only=True))
