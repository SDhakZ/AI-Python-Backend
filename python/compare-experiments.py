import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Paths to the files produced by your two training scripts ---
XGB_SUMMARY = "summary_xgb_only_base.csv"
ENS_SUMMARY = "summary_ensemble_popkey.csv"

# Paths to reports
XGB_VAL_REPORT = "val_report_xgb_only_base.txt"
XGB_CLS_REPORT = "classified_report_xgb_only_base.txt"
ENS_VAL_REPORT = "val_report_ensemble_popkey.txt"
ENS_CLS_REPORT = "classified_report_ensemble_popkey.txt"


def read_summaries(xgb_csv: str, ens_csv: str) -> pd.DataFrame:
    sx = pd.read_csv(xgb_csv)
    se = pd.read_csv(ens_csv)
    df = pd.DataFrame({
        "Model": ["XGBoost Base", "Ensemble + POP+KEY"],
        "Val_Accuracy": [sx.loc[0, "Val_Accuracy"], se.loc[0, "Val_Accuracy"]],
        "Val_F1_weighted": [sx.loc[0, "Val_F1_weighted"], se.loc[0, "Val_F1_weighted"]],
        "Cls_Accuracy": [sx.loc[0, "Cls_Accuracy"], se.loc[0, "Cls_Accuracy"]],
        "Cls_F1_weighted": [sx.loc[0, "Cls_F1_weighted"], se.loc[0, "Cls_F1_weighted"]],
    })
    return df


def parse_weighted_pr_re_f1(report_path: str) -> dict:
    text = Path(report_path).read_text(encoding="utf-8")
    m = re.search(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    precision, recall, f1 = map(float, m.groups())
    return {"precision_w": precision, "recall_w": recall, "f1_w": f1}


# Read CSV summaries
acc_f1_df = read_summaries(XGB_SUMMARY, ENS_SUMMARY)

# Parse weighted precision/recall
xgb_val = parse_weighted_pr_re_f1(XGB_VAL_REPORT)
xgb_cls = parse_weighted_pr_re_f1(XGB_CLS_REPORT)
ens_val = parse_weighted_pr_re_f1(ENS_VAL_REPORT)
ens_cls = parse_weighted_pr_re_f1(ENS_CLS_REPORT)

# Validation metrics
val_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (W)", "Recall (W)", "F1-score (W)"],
    "XGBoost Base": [
        acc_f1_df.loc[0, "Val_Accuracy"],
        xgb_val["precision_w"],
        xgb_val["recall_w"],
        acc_f1_df.loc[0, "Val_F1_weighted"],
    ],
    "Ensemble + POP+KEY": [
        acc_f1_df.loc[1, "Val_Accuracy"],
        ens_val["precision_w"],
        ens_val["recall_w"],
        acc_f1_df.loc[1, "Val_F1_weighted"],
    ],
})

# Classified metrics
cls_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (W)", "Recall (W)", "F1-score (W)"],
    "XGBoost Base": [
        acc_f1_df.loc[0, "Cls_Accuracy"],
        xgb_cls["precision_w"],
        xgb_cls["recall_w"],
        acc_f1_df.loc[0, "Cls_F1_weighted"],
    ],
    "Ensemble + POP+KEY": [
        acc_f1_df.loc[1, "Cls_Accuracy"],
        ens_cls["precision_w"],
        ens_cls["recall_w"],
        acc_f1_df.loc[1, "Cls_F1_weighted"],
    ],
})


def plot_grouped_bars_with_values(df: pd.DataFrame, title: str, out_png: str):
    metrics = df["Metric"].tolist()
    m1 = df["XGBoost Base"].values
    m2 = df["Ensemble + POP+KEY"].values

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - width/2, m1, width, label="XGBoost Base")
    bars2 = plt.bar(x + width/2, m2, width, label="Ensemble + POP+KEY")

    # Add text labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}",
                 ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(x, metrics, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


val_plot_path = plot_grouped_bars_with_values(val_df, "Validation Set: Metric Comparison", "comparison_validation.png")
cls_plot_path = plot_grouped_bars_with_values(cls_df, "Classified Set: Metric Comparison", "comparison_classified.png")

(val_plot_path, cls_plot_path)
print(val_plot_path, cls_plot_path)


import matplotlib.pyplot as plt
import numpy as np

# Original XGBoost values
xgb_values = [0.475, 0.444, 0.475, 0.437]

# Updated Ensemble values
ensemble_values = [0.542, 0.538, 0.534, 0.534]

# Metric names
metrics = ["Accuracy", "Precision (W)", "Recall (W)", "F1-score (W)"]

# Bar chart parameters
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, xgb_values, width, label="XGBoost Base", color="#1f77b4")
bars2 = ax.bar(x + width/2, ensemble_values, width, label="Ensemble + POP+KEY", color="#ff7f0e")

# Adding values above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Labels and formatting
ax.set_ylabel("Score")
ax.set_title("Validation Set: Metric Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 0.65)
ax.legend()

plt.tight_layout()
plt.show()
