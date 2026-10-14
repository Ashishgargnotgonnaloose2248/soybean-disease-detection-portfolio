import os
import torch
import timm
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# ==========================
# CONFIG
# ==========================

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

BATCH_SIZE = 16
NUM_CLASSES = 6
EPOCHS = 25
LR = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==========================
# TRANSFORMS
# ==========================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.05
    ),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# DATASETS
# ==========================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClasses:")
print(train_dataset.classes)

# ==========================
# CLASS WEIGHTS
# ==========================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass Weights:")
print(class_weights)

# ==========================
# DATALOADERS
# ==========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ==========================
# MODEL
# ==========================

model = timm.create_model(
    "efficientformer_l3",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

# ==========================
# LOSS
# ==========================

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

# ==========================
# OPTIMIZER
# ==========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# ==========================
# SCHEDULER
# ==========================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ==========================
# TRAINING
# ==========================

best_acc = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100.0 * correct / total

    scheduler.step(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}% | "
        f"LR: {current_lr:.6f}"
    )

    # ==========================
    # SAVE BEST MODEL
    # ==========================

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "checkpoints/best_model_v2.pth"
        )

        print(
            f"New Best Model Saved! "
            f"Accuracy = {val_acc:.2f}%"
        )

print("\n===================================")
print("Training Complete")
print(f"Best Validation Accuracy: {best_acc:.2f}%")
print("===================================")