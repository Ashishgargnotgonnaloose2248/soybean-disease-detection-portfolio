import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "efficientformer_l1"

CHECKPOINT = "checkpoints/best_model_v3.pth"

TEST_DIR = "dataset_split/test"      # <-- Change if needed

OUTPUT_DIR = "outputs/graphs"

NUM_CLASSES = 6

BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "Caterpillar_and_Semilooper_Pest_Attack",
    "Frog_Leaf_Eye",
    "Healthy",
    "Mosaic",
    "Rust",
    "Spectoria_Brown_Spot"
]

# ==========================================================
# MODEL
# ==========================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ==========================================================
# DATASET
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes Found:")
print(test_dataset.classes)

# ==========================================================
# INFERENCE
# ==========================================================

all_labels = []
all_predictions = []
all_confidences = []

print("\nRunning inference...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_confidences = np.array(all_confidences)

print("Inference Complete.")
print("Total Images :", len(all_labels))

# ==========================================================
# CONFIDENCE THRESHOLDS
# ==========================================================

thresholds = np.linspace(
    0.0,
    1.0,
    101
)

precision_values = []
recall_values = []
f1_values = []

# ==========================================================
# COMPUTE CURVES
# ==========================================================

print("\nGenerating Confidence Curves...")

for threshold in thresholds:

    valid = all_confidences >= threshold

    if np.sum(valid) == 0:

        precision_values.append(np.nan)
        recall_values.append(np.nan)
        f1_values.append(np.nan)

        continue

    y_true = all_labels[valid]
    y_pred = all_predictions[valid]

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

precision_values = np.array(precision_values)
recall_values = np.array(recall_values)
f1_values = np.array(f1_values)

# ==========================================================
# BEST F1
# ==========================================================

best_idx = np.nanargmax(f1_values)

best_threshold = thresholds[best_idx]
best_f1 = f1_values[best_idx]

print(f"\nBest Threshold : {best_threshold:.2f}")
print(f"Best F1 Score  : {best_f1:.4f}")

# ==========================================================
# PLOT
# ==========================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(18,5),
    dpi=300
)

# -----------------------------
# F1
# -----------------------------

axs[0].plot(
    thresholds,
    f1_values,
    color="blue",
    linewidth=3
)

axs[0].axvline(
    best_threshold,
    color="red",
    linestyle="--",
    linewidth=2
)

axs[0].scatter(
    best_threshold,
    best_f1,
    color="red",
    s=60,
    zorder=5
)

axs[0].set_title(
    "F1-Confidence Curve",
    fontsize=14,
    fontweight="bold"
)

axs[0].set_xlabel("Confidence")
axs[0].set_ylabel("F1 Score")

axs[0].grid(alpha=0.3)

# -----------------------------
# Precision
# -----------------------------

axs[1].plot(
    thresholds,
    precision_values,
    color="green",
    linewidth=3
)

axs[1].set_title(
    "Precision-Confidence Curve",
    fontsize=14,
    fontweight="bold"
)

axs[1].set_xlabel("Confidence")
axs[1].set_ylabel("Precision")

axs[1].grid(alpha=0.3)

# -----------------------------
# Recall
# -----------------------------

axs[2].plot(
    thresholds,
    recall_values,
    color="darkorange",
    linewidth=3
)

axs[2].set_title(
    "Recall-Confidence Curve",
    fontsize=14,
    fontweight="bold"
)

axs[2].set_xlabel("Confidence")
axs[2].set_ylabel("Recall")

axs[2].grid(alpha=0.3)

# ==========================================================
# SAVE
# ==========================================================

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_DIR,
    "confidence_curves.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "confidence_curves.pdf"
)

plt.savefig(
    png_path,
    dpi=800,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# DONE
# ==========================================================

print("\n========================================")
print("Confidence Curves Saved Successfully")
print("========================================")

print("\nSaved Files:")
print(png_path)
print(pdf_path)