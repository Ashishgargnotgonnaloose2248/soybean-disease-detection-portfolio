import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================

METRICS_FILE = "outputs/metrics/all_models_comparison.csv"

OUTPUT_DIR = "outputs/comparison_graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================

df = pd.read_csv(METRICS_FILE)

print("\nLoaded Model Results:")
print(df)

# ==========================================================
# MODEL ORDER
# ==========================================================

MODEL_ORDER = [
    "EfficientFormer-L1",
    "VGG16",
    "VGG19",
    "ResNet50",
    "InceptionV3"
]

df["model"] = pd.Categorical(
    df["model"],
    categories=MODEL_ORDER,
    ordered=True
)

df = df.sort_values("model").reset_index(drop=True)

models = df["model"].astype(str).tolist()

print("\nComparison Order:")
for model in models:
    print(model)

# ==========================================================
# COMMON GRAPH SETTINGS
# ==========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

# ==========================================================
# 1. ACCURACY COMPARISON
# ==========================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    df["accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Test Accuracy Comparison of Classification Models"
)

plt.xticks(rotation=20)

plt.ylim(
    max(0, df["accuracy"].min() - 10),
    100
)

for bar, value in zip(
    bars,
    df["accuracy"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# 2. PRECISION / RECALL / F1 COMPARISON
# ==========================================================

x = np.arange(len(models))

width = 0.25

plt.figure(figsize=(11, 6))

precision_bars = plt.bar(
    x - width,
    df["precision"],
    width,
    label="Precision"
)

recall_bars = plt.bar(
    x,
    df["recall"],
    width,
    label="Recall"
)

f1_bars = plt.bar(
    x + width,
    df["f1_score"],
    width,
    label="F1-score"
)

plt.xlabel("Model")
plt.ylabel("Score (%)")

plt.title(
    "Precision, Recall and F1-score Comparison"
)

plt.xticks(
    x,
    models,
    rotation=20
)

plt.ylim(
    max(
        0,
        min(
            df["precision"].min(),
            df["recall"].min(),
            df["f1_score"].min()
        ) - 10
    ),
    100
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_f1_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_f1_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# 3. ROC-AUC / PR-AUC COMPARISON
# ==========================================================

plt.figure(figsize=(10, 6))

roc_bars = plt.bar(
    x - width / 2,
    df["roc_auc"],
    width,
    label="ROC-AUC"
)

pr_bars = plt.bar(
    x + width / 2,
    df["pr_auc"],
    width,
    label="PR-AUC"
)

plt.xlabel("Model")
plt.ylabel("AUC Score")

plt.title(
    "ROC-AUC and PR-AUC Comparison"
)

plt.xticks(
    x,
    models,
    rotation=20
)

plt.ylim(0, 1.05)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "auc_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "auc_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# 4. PARAMETER COUNT COMPARISON
# ==========================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    df["parameters_million"]
)

plt.xlabel("Model")
plt.ylabel("Parameters (Millions)")

plt.title(
    "Model Parameter Count Comparison"
)

plt.xticks(
    rotation=20
)

for bar, value in zip(
    bars,
    df["parameters_million"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{value:.2f}M",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "parameters_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "parameters_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 5. INFERENCE TIME COMPARISON
# ==========================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    df["inference_time_ms"]
)

plt.xlabel("Model")
plt.ylabel("Inference Time (ms/image)")

plt.title(
    "Inference Time Comparison"
)

plt.xticks(
    rotation=20
)

for bar, value in zip(
    bars,
    df["inference_time_ms"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "inference_time_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "inference_time_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 6. PARAMETERS VS ACCURACY
# ==========================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["parameters_million"],
    df["accuracy"],
    s=120
)

for _, row in df.iterrows():

    plt.annotate(
        row["model"],
        (
            row["parameters_million"],
            row["accuracy"]
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=10
    )

plt.xlabel("Parameters (Millions)")

plt.ylabel("Test Accuracy (%)")

plt.title(
    "Model Complexity vs Test Accuracy"
)

plt.grid(
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "parameters_vs_accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "parameters_vs_accuracy.pdf"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 7. INFERENCE TIME VS ACCURACY
# ==========================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["inference_time_ms"],
    df["accuracy"],
    s=120
)

for _, row in df.iterrows():

    plt.annotate(
        row["model"],
        (
            row["inference_time_ms"],
            row["accuracy"]
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=10
    )

plt.xlabel(
    "Inference Time (ms/image)"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.title(
    "Inference Speed vs Test Accuracy"
)

plt.grid(
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "inference_time_vs_accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "inference_time_vs_accuracy.pdf"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 8. FINAL COMPARISON TABLE
# ==========================================================

comparison_table = df.copy()

comparison_table = comparison_table.rename(
    columns={
        "model": "Model",
        "accuracy": "Accuracy (%)",
        "precision": "Precision (%)",
        "recall": "Recall (%)",
        "f1_score": "F1-score (%)",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "parameters_million": "Parameters (M)",
        "inference_time_ms": "Inference (ms/image)"
    }
)

comparison_table = comparison_table[
    [
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1-score (%)",
        "ROC-AUC",
        "PR-AUC",
        "Parameters (M)",
        "Inference (ms/image)"
    ]
]

# Round values for clean table

comparison_table["Accuracy (%)"] = (
    comparison_table["Accuracy (%)"].round(2)
)

comparison_table["Precision (%)"] = (
    comparison_table["Precision (%)"].round(2)
)

comparison_table["Recall (%)"] = (
    comparison_table["Recall (%)"].round(2)
)

comparison_table["F1-score (%)"] = (
    comparison_table["F1-score (%)"].round(2)
)

comparison_table["ROC-AUC"] = (
    comparison_table["ROC-AUC"].round(4)
)

comparison_table["PR-AUC"] = (
    comparison_table["PR-AUC"].round(4)
)

comparison_table["Parameters (M)"] = (
    comparison_table["Parameters (M)"].round(2)
)

comparison_table["Inference (ms/image)"] = (
    comparison_table["Inference (ms/image)"].round(2)
)


# ==========================================================
# SAVE TABLE
# ==========================================================

table_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_table.csv"
)

comparison_table.to_csv(
    table_path,
    index=False
)

print(
    "\nComparison table saved:",
    table_path
)


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

print("\n" + "=" * 90)

print(
    "MODEL COMPARISON"
)

print("=" * 90)

print(
    comparison_table.to_string(
        index=False
    )
)

print("=" * 90)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

print(
    "\nAll comparison graphs saved to:"
)

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

for file in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        " -",
        file
    )

print("\nComparison completed successfully.")