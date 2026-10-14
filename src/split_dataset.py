import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ==========================
# CONFIG
# ==========================

DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

# ==========================
# REMOVE OLD SPLIT
# ==========================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# ==========================
# CREATE NEW FOLDERS
# ==========================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ==========================
# SPLIT EACH CLASS
# ==========================

classes = sorted(os.listdir(DATASET_DIR))

print("=" * 60)
print("Creating Train / Validation / Test Split")
print("=" * 60)

for class_name in classes:

    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

    for split_name, split_images in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):

        destination = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        os.makedirs(destination, exist_ok=True)

        for img in split_images:

            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(destination, img)
            )

    print(f"{class_name}")
    print(f"  Train : {len(train_imgs)}")
    print(f"  Val   : {len(val_imgs)}")
    print(f"  Test  : {len(test_imgs)}")
    print()

print("=" * 60)
print("Dataset Split Completed Successfully!")
print("=" * 60)