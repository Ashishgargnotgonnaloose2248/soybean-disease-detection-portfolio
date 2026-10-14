import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

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
# TRANSFORM
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
# ROC CURVE
# ==========================================================

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):

    fpr[i], tpr[i], _ = roc_curve(
        labels_bin[:, i],
        all_probs[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(10,8))

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
        fpr[i],
        tpr[i],
        color=color,
        linewidth=2,
        label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})"
    )

# Random classifier
plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="black",
    linewidth=1.5,
    label="Random Classifier"
)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)

plt.title(
    "Multi-class ROC Curve - EfficientFormer-L1",
    fontsize=14,
    fontweight="bold"
)

plt.legend(
    loc="lower right",
    fontsize=9
)

plt.grid(alpha=0.3)

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    "roc_curve.png"
)

plt.savefig(
    save_path,
    dpi=600
)

plt.show()

print("\n========================================")
print("ROC Curve Saved Successfully")
print("========================================")
print(save_path)

print("\nAUC Scores")
print("----------------------------------------")

for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:35s} : {roc_auc[i]:.4f}"
    )

print("----------------------------------------")
print(
    f"Mean AUC : {np.mean(list(roc_auc.values())):.4f}"
)