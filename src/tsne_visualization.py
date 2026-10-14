import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

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
# FEATURE EXTRACTION
# ==========================================================

features = []

labels = []

with torch.no_grad():

    for images, target in test_loader:

        images = images.to(DEVICE)

        # 768-dimensional feature vector
        embedding = model.forward_features(images)

        # Global Pooling
        if embedding.ndim == 4:
            embedding = embedding.mean(dim=[2,3])

        elif embedding.ndim == 3:
            embedding = embedding.mean(dim=1)

        features.append(
            embedding.cpu().numpy()
        )

        labels.extend(
            target.numpy()
        )

features = np.concatenate(features, axis=0)

labels = np.array(labels)

print("\nFeature Shape :", features.shape)
print("Total Images  :", len(labels))

# ==========================================================
# t-SNE
# ==========================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

features_2d = tsne.fit_transform(features)

print("t-SNE Completed.")

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

for i in range(NUM_CLASSES):

    idx = labels == i

    plt.scatter(
        features_2d[idx,0],
        features_2d[idx,1],
        s=45,
        alpha=0.75,
        color=colors[i],
        label=CLASS_NAMES[i]
    )

plt.title(
    "t-SNE Visualization of Deep Feature Embeddings",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "t-SNE Dimension 1",
    fontsize=13
)

plt.ylabel(
    "t-SNE Dimension 2",
    fontsize=13
)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.grid(alpha=0.25)

plt.legend(
    fontsize=10,
    loc="best",
    frameon=True
)

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_DIR,
    "tsne_visualization.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "tsne_visualization.pdf"
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
# FINISHED
# ==========================================================

print("\n========================================")
print("t-SNE Visualization Saved Successfully")
print("========================================")

print("\nSaved Files:")
print(png_path)
print(pdf_path)