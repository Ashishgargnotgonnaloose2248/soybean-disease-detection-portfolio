import os
import cv2
import torch
import timm
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "efficientformer_l1"
CHECKPOINT = "checkpoints/best_model_v3.pth"

GRADCAM_DIR = "gradcam_images/set_2"

OUTPUT_DIR = "gradcam_results"

NUM_CLASSES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# IMAGE TRANSFORM
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
# RESHAPE TRANSFORM
# ==========================================================

def reshape_transform(tensor):

    # tensor shape = (B, 49, 448)

    if len(tensor.shape) == 3:

        result = tensor.reshape(
            tensor.size(0),
            7,
            7,
            tensor.size(2)
        )

        result = result.permute(
            0,
            3,
            1,
            2
        )

        return result

    elif len(tensor.shape) == 4:

        return tensor

    else:

        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

# ==========================================================
# TARGET LAYER
# ==========================================================

target_layers = [
    model.stages[3].blocks[4]
]

# ==========================================================
# CREATE CAM
# ==========================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

# ==========================================================
# CREATE FIGURE (IEEE STYLE)
# ==========================================================

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(22, 12))

gs = gridspec.GridSpec(
    4,
    6,
    height_ratios=[0.18, 1, 0.18, 1],
    hspace=0.35,
    wspace=0.15
)

fig.suptitle(
    "Grad-CAM Visualization of EfficientFormer-L1",
    fontsize=22,
    fontweight="bold",
    y=0.98
)

# Short display names
DISPLAY_NAMES = {
    "Caterpillar_and_Semilooper_Pest_Attack":
        "Caterpillar Attack",

    "Frog_Leaf_Eye":
        "Frog Leaf Eye",

    "Healthy":
        "Healthy",

    "Mosaic":
        "Mosaic",

    "Rust":
        "Rust",

    "Spectoria_Brown_Spot":
        "Septoria Brown Spot"
}

# Keeps track of subplot position
plot_number = 0


def add_border(ax):

    for spine in ax.spines.values():

        spine.set_visible(True)

        spine.set_linewidth(1.2)

        spine.set_edgecolor("black")

    ax.set_xticks([])
    ax.set_yticks([])


def get_subplot_indices(index):
    """
    Returns row/column positions.

    Layout:

    Caterpillar | Frog | Healthy
    Mosaic      | Rust | Septoria
    """

    if index < 3:

        row = 1

        col = index * 2

    else:

        row = 3

        col = (index - 3) * 2

    return row, col

plot_index = 1

# ==========================================================
# PROCESS EACH CLASS
# ==========================================================

IMAGE_FILES = {
    "Caterpillar_and_Semilooper_Pest_Attack": "caterpillar.jpg",
    "Frog_Leaf_Eye": "frog_leaf_eye.jpg",
    "Healthy": "healthy.jpg",
    "Mosaic": "mosaic.jpg",
    "Rust": "rust.jpg",
    "Spectoria_Brown_Spot": "septoria.jpg"
}

for idx, class_name in enumerate(CLASS_NAMES):

    image_name = IMAGE_FILES[class_name]

    image_path = os.path.join(GRADCAM_DIR, image_name)

    print(f"\nProcessing : {class_name}")
    print(f"Image : {image_name}")

    # ---------------------------
    # Load image
    # ---------------------------

    pil_image = Image.open(image_path).convert("RGB")

    pil_image = pil_image.resize((224,224))

    rgb_img = np.array(pil_image).astype(np.float32)/255.0

    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # ---------------------------
    # Prediction
    # ---------------------------

    with torch.no_grad():

        output = model(input_tensor)

        probs = torch.softmax(output,dim=1)

        pred = torch.argmax(probs).item()

        conf = probs[0][pred].item()

    print(f"Prediction : {CLASS_NAMES[pred]} ({conf*100:.2f}%)")

    # ---------------------------
    # GradCAM
    # ---------------------------

    targets = [
        ClassifierOutputTarget(pred)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # ---------------------------
    # Position
    # ---------------------------

    row, col = get_subplot_indices(idx)

    # ===========================
    # ORIGINAL IMAGE
    # ===========================

    ax1 = fig.add_subplot(gs[row, col])

    ax1.imshow(rgb_img)

    ax1.set_title(
        "Original",
        fontsize=11,
        fontweight="bold",
        pad=8
    )

    add_border(ax1)

    # ===========================
    # GRADCAM
    # ===========================

    ax2 = fig.add_subplot(gs[row, col+1])

    ax2.imshow(visualization)

    ax2.set_title(
        "Grad-CAM",
        fontsize=11,
        fontweight="bold",
        pad=8
    )

    add_border(ax2)

    # ===========================
    # CLASS LABEL
    # ===========================

    x = (ax1.get_position().x0 + ax2.get_position().x1)/2

    y = ax1.get_position().y1 + 0.03

    fig.text(
        x,
        y,
        DISPLAY_NAMES[class_name],
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    # ===========================
    # Prediction
    # ===========================

    y2 = ax1.get_position().y0 - 0.035

    fig.text(
        x,
        y2,
        f"Pred: {DISPLAY_NAMES[CLASS_NAMES[pred]]}\nConfidence: {conf*100:.2f}%",
        ha="center",
        fontsize=10
    )

# ==========================================================
# SAVE
# ==========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_path = os.path.join(
    OUTPUT_DIR,
    f"gradcam_{timestamp}.png"
)

plt.savefig(
    save_path,
    dpi=800,
    bbox_inches="tight",
    facecolor="white"
)

print(f"\nSaved to: {save_path}")

plt.show()

print("\n=======================================")
print("GradCAM Figure Saved Successfully")
print("=======================================")