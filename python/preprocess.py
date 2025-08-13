# preprocess_datasets.py
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DUR_COL = "duration_in min/ms"
TARGET = "Class"

def common_preclean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix duration units: values < 30 are minutes -> convert to ms."""
    out = df.copy()
    if DUR_COL in out.columns:
        mask = out[DUR_COL] < 30
        out.loc[mask, DUR_COL] = out.loc[mask, DUR_COL] * 60000
    return out

def cyclic_encode_key(key_series: pd.Series, n_cycle: int = 12) -> pd.DataFrame:
    """Encode musical key (0..11) into sin/cos; invalid/missing -> NaN."""
    k = key_series.copy()
    k = k.where((k >= 0) & (k < n_cycle))
    angle = 2 * np.pi * (k / n_cycle)
    return pd.DataFrame({"key_sin": np.sin(angle), "key_cos": np.cos(angle)})

# --- Load raw ---
train_raw = pd.read_csv("train.csv")
classified_raw = pd.read_csv("classified.csv")

# Drop unknown classes from classified
if TARGET in classified_raw.columns:
    classified_raw = classified_raw[classified_raw[TARGET] != -1].copy()

# Helper to drop columns if present (handles case correctly)
def drop_if_present(df: pd.DataFrame, cols):
    cols_to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=cols_to_drop)

# ================= DATASET A (BASE) =================
# Drop: Artist Name, Track Name, Popularity, energy  
drop_cols_base = ["Artist Name", "Track Name", "Popularity"]
train_base = drop_if_present(train_raw, drop_cols_base)
cls_base   = drop_if_present(classified_raw, drop_cols_base)

train_base = common_preclean(train_base)
cls_base   = common_preclean(cls_base)

# Fill instrumentalness with TRAIN mean
if "instrumentalness" in train_base.columns:
    instr_mean_base = train_base["instrumentalness"].mean()
    for df in (train_base, cls_base):
        df["instrumentalness"] = df["instrumentalness"].fillna(instr_mean_base)

# Raw key with -1 for missing
if "key" in train_base.columns:
    for df in (train_base, cls_base):
        df["key"] = df["key"].fillna(-1)

# Targeted skewness fix (log1p) for BASE
for df in (train_base, cls_base):
    for col in ["liveness", "instrumentalness"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

# ================= DATASET B (POP+KEY) =================
# Keep Popularity; drop only names and energy
drop_cols_popkey = ["Artist Name", "Track Name", "energy"]
train_pk = drop_if_present(train_raw, drop_cols_popkey)
cls_pk   = drop_if_present(classified_raw, drop_cols_popkey)

train_pk = common_preclean(train_pk)
cls_pk   = common_preclean(cls_pk)

# Fill instrumentalness with TRAIN mean
if "instrumentalness" in train_pk.columns:
    instr_mean_pk = train_pk["instrumentalness"].mean()
    for df in (train_pk, cls_pk):
        df["instrumentalness"] = df["instrumentalness"].fillna(instr_mean_pk)

# Cyclic encode key; drop original key
if "key" in train_pk.columns:
    train_key_enc = cyclic_encode_key(train_pk["key"])
    cls_key_enc   = cyclic_encode_key(cls_pk["key"])
    train_pk = pd.concat([train_pk.drop(columns=["key"]), train_key_enc], axis=1)
    cls_pk   = pd.concat([cls_pk.drop(columns=["key"]), cls_key_enc], axis=1)

# Targeted skewness fix (log1p) for POP+KEY
for df in (train_pk, cls_pk):
    for col in ["liveness", "instrumentalness"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

# --- Save AFTER all transforms ---
train_base.to_csv("train_preprocessed.csv", index=False)
cls_base.to_csv("classified_preprocessed.csv", index=False)

train_pk.to_csv("train_preprocessed_popkey.csv", index=False)
cls_pk.to_csv("classified_preprocessed_popkey.csv", index=False)

print("Saved:")
print(" - BASE:     train_preprocessed.csv, classified_preprocessed.csv")
print(" - POP+KEY:  train_preprocessed_popkey.csv, classified_preprocessed_popkey.csv")
