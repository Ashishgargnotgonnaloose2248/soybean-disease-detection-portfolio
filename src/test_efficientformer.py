import timm
import torch

model = timm.create_model(
    "efficientformer_l1",
    pretrained=True,
    num_classes=6
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Model Loaded Successfully")
print("Device:", device)