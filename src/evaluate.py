import os
import torch
import timm
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# CONFIG
# ==========================

TEST_DIR = "dataset_split/test"
MODEL_PATH = "checkpoints/best_model_v3.pth"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================
# TRANSFORMS
# ==========================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# DATASET
# ==========================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("\nClasses:")
print(class_names)

# ==========================
# MODEL
# ==========================

model = timm.create_model(
    "efficientformer_l1",
    pretrained=False,
    num_classes=6
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("\nLoaded:", MODEL_PATH)

# ==========================
# PREDICTIONS
# ==========================

all_labels = []
all_preds = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ==========================
# CLASSIFICATION REPORT
# ==========================

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("\nClassification report saved.")

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor="black",
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size":10}
)

plt.title(
    "Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Label",
    fontsize=12
)

plt.ylabel(
    "True Label",
    fontsize=12
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix saved successfully.")
print("Location :", os.path.join(OUTPUT_DIR, "confusion_matrix.png"))