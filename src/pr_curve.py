import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# ==========================================================
# CONFIG
# ==========================================================

TEST_DIR = "dataset_split/test"

MODEL_NAME = "efficientformer_l1"

CHECKPOINT = "checkpoints/best_model_v3.pth"

OUTPUT_DIR = "outputs/graphs"

NUM_CLASSES = 6

BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# TRANSFORMS
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# DATASET
# ==========================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

CLASS_NAMES = test_dataset.classes

print("Classes:")
print(CLASS_NAMES)

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
# COLLECT PREDICTIONS
# ==========================================================

all_labels = []

all_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.numpy())

        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)

all_probs = np.array(all_probs)

# ==========================================================
# BINARIZE LABELS
# ==========================================================

labels_bin = label_binarize(
    all_labels,
    classes=np.arange(NUM_CLASSES)
)
# ==========================================================
# PRECISION-RECALL CURVE
# ==========================================================

precision = dict()
recall = dict()
average_precision = dict()

for i in range(NUM_CLASSES):

    precision[i], recall[i], _ = precision_recall_curve(
        labels_bin[:, i],
        all_probs[:, i]
    )

    average_precision[i] = average_precision_score(
        labels_bin[:, i],
        all_probs[:, i]
    )

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(11,9))

colors = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown"
]

for i, color in enumerate(colors):

    plt.plot(
        recall[i],
        precision[i],
        color=color,
        linewidth=3,
        label=f"{CLASS_NAMES[i]} (AP = {average_precision[i]:.3f})"
    )

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])

plt.xlabel(
    "Recall",
    fontsize=14
)

plt.ylabel(
    "Precision",
    fontsize=14
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title(
    "Precision-Recall Curve",
    fontsize=16,
    fontweight="bold"
)

plt.grid(alpha=0.3)

plt.legend(
    loc="lower left",
    fontsize=11
)

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_DIR,
    "pr_curve.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "pr_curve.pdf"
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
# RESULTS
# ==========================================================

print("\n========================================")
print("Precision-Recall Curve Saved Successfully")
print("========================================")

print("\nSaved Files:")
print(png_path)
print(pdf_path)

print("\nAverage Precision Scores")
print("----------------------------------------")

for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:35s} : {average_precision[i]:.4f}"
    )

print("----------------------------------------")
print(
    f"Mean Average Precision (mAP): {np.mean(list(average_precision.values())):.4f}"
)