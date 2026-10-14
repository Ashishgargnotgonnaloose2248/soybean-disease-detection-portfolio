import os
import cv2
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "efficientformer_l1"

CHECKPOINT = "checkpoints/best_model_v3.pth"

IMAGE_PATH = "dataset_split/test/Frog_Leaf_Eye/frogeye_7.jpg"      # <-- Change to your image

OUTPUT_DIR = "outputs/feature_maps"

NUM_CLASSES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# IMAGE
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

pil_image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

# Original image for display
original_image = np.array(
    pil_image.resize((224,224))
)

# ==========================================================
# FEATURE MAP STORAGE
# ==========================================================

activations = {}

def get_activation(name):

    def hook(model, input, output):

        activations[name] = output.detach().cpu()

    return hook

# ==========================================================
# REGISTER HOOKS
# ==========================================================

model.stem.register_forward_hook(
    get_activation("Stem")
)

model.stages[0].register_forward_hook(
    get_activation("Stage1")
)

model.stages[1].register_forward_hook(
    get_activation("Stage2")
)

model.stages[2].register_forward_hook(
    get_activation("Stage3")
)

model.stages[3].register_forward_hook(
    get_activation("Stage4")
)

# ==========================================================
# FORWARD PASS
# ==========================================================

with torch.no_grad():

    output = model(input_tensor)

print("\nCaptured Feature Maps:\n")

for key, value in activations.items():

    print(key, value.shape)

# ==========================================================
# PLOT FEATURE MAPS
# ==========================================================

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18,14), dpi=300)

gs = GridSpec(
    17,
    6,
    figure=fig,
    hspace=0.35,
    wspace=0.15
)

# ----------------------------------------------------------
# Original Image
# ----------------------------------------------------------

ax = fig.add_subplot(gs[:,0])

ax.imshow(original_image)

ax.set_title(
    "Original Image",
    fontsize=16,
    fontweight="bold"
)

ax.axis("off")

# ----------------------------------------------------------
# Plot Helper
# ----------------------------------------------------------

stage_names = [
    "Stem",
    "Stage1",
    "Stage2",
    "Stage3",
    "Stage4"
]

for col, stage in enumerate(stage_names, start=1):

    fmap = activations[stage]

    # Stage4 -> Tokens to Feature Maps
    if fmap.ndim == 3:

        fmap = fmap.squeeze(0)

        fmap = fmap.reshape(7,7,448)

        fmap = fmap.permute(2,0,1)

    else:

        fmap = fmap.squeeze(0)

    num_maps = min(16, fmap.shape[0])

    # Stage Title
    title_ax = fig.add_subplot(gs[0,col])

    title_ax.set_title(
        stage,
        fontsize=15,
        fontweight="bold"
    )

    title_ax.axis("off")

    for i in range(num_maps):

        row = (i // 4) * 4 + 1

        subcol = i % 4

        ax = fig.add_subplot(
            gs[
                row:row+4,
                col
            ]
        )

        feature = fmap[i].numpy()

        feature -= feature.min()

        feature /= feature.max() + 1e-8

        ax.imshow(
            feature,
            cmap="viridis"
        )

        ax.axis("off")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

plt.suptitle(
    "Layer-wise Feature Maps of EfficientFormer-L1",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_DIR,
    "feature_maps.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "feature_maps.pdf"
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

print("\nSaved:")
print(png_path)
print(pdf_path)    