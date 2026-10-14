import os
import torch
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =====================
# CONFIG
# =====================

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

BATCH_SIZE = 16
NUM_CLASSES = 6
EPOCHS = 20
LR = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================
# TRANSFORMS
# =====================

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================
# DATASET
# =====================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:")
print(train_dataset.classes)

# =====================
# MODEL
# =====================

model = timm.create_model(
    "efficientformer_l1",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

# =====================
# LOSS
# =====================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

# =====================
# TRAIN LOOP
# =====================

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    # ------------------
    # VALIDATION
    # ------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        print(
            f"Best model saved! "
            f"Accuracy = {val_acc:.2f}%"
        )

print(
    f"\nTraining Complete\n"
    f"Best Validation Accuracy: {best_acc:.2f}%"
)