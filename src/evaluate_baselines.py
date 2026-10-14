import os
import json
import time

import torch
import timm
import numpy as np
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# ==========================================================
# CONFIG
# ==========================================================

TEST_DIR = "dataset_split/test"

NUM_CLASSES = 6
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

os.makedirs(
    "outputs/metrics",
    exist_ok=True
)

os.makedirs(
    "outputs/evaluation_reports",
    exist_ok=True
)

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "Caterpillar_and_Semilooper_Pest_Attack",
    "Frog_Leaf_Eye",
    "Healthy",
    "Mosaic",
    "Rust",
    "Spectoria_Brown_Spot"
]

# ==========================================================
# MODELS
# ==========================================================

MODELS = {
    "VGG16": {
        "model_name": "vgg16",
        "checkpoint": "checkpoints/vgg16_best.pth",
        "image_size": 224
    },

    "EfficientFormer-L1": {
        "model_name": "efficientformer_l1",
        "checkpoint": "checkpoints/efficientformer_l1_best.pth",
        "image_size": 224
    },

    "VGG19": {
        "model_name": "vgg19",
        "checkpoint": "checkpoints/vgg19_best.pth",
        "image_size": 224
    },

    "ResNet50": {
        "model_name": "resnet50",
        "checkpoint": "checkpoints/resnet50_best.pth",
        "image_size": 224
    },

    "InceptionV3": {
        "model_name": "inception_v3",
        "checkpoint": "checkpoints/inception_v3_best.pth",
        "image_size": 299
    }
}

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    display_name,
    model_name,
    checkpoint_path,
    image_size
):

    print("\n" + "=" * 65)

    print("Evaluating:", display_name)

    print("Checkpoint:", checkpoint_path)

    print("=" * 65)

    # ------------------------------------------------------
    # CHECK CHECKPOINT
    # ------------------------------------------------------

    if not os.path.exists(checkpoint_path):

        print(
            f"\nCheckpoint not found:\n"
            f"{checkpoint_path}"
        )

        return None

    # ------------------------------------------------------
    # TEST TRANSFORM
    # ------------------------------------------------------

    test_transform = transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    # ------------------------------------------------------
    # TEST DATASET
    # ------------------------------------------------------

    test_dataset = datasets.ImageFolder(

        TEST_DIR,

        transform=test_transform
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True
    )

    print(
        "Test Images:",
        len(test_dataset)
    )

    print(
        "Classes:",
        test_dataset.classes
    )

    # ------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------

    if "efficientformer" in model_name.lower():

        model = timm.create_model(

            model_name,

            pretrained=False,

            num_classes=NUM_CLASSES,

            drop_rate=0.3,

            drop_path_rate=0.1
        )

    else:

        model = timm.create_model(

            model_name,

            pretrained=False,

            num_classes=NUM_CLASSES
        )

    # ------------------------------------------------------
    # LOAD CHECKPOINT
    # ------------------------------------------------------

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)

    model.eval()

    # ------------------------------------------------------
    # PARAMETER COUNT
    # ------------------------------------------------------

    total_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()
    )

    trainable_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad
    )

    print(

        "Total Parameters:",

        f"{total_parameters:,}"
    )

    # ------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------

    all_labels = []

    all_predictions = []

    all_probabilities = []

    total_inference_time = 0.0

    total_images = 0

    # ------------------------------------------------------
    # TESTING
    # ------------------------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(

                DEVICE,

                non_blocking=True
            )

            labels = labels.to(

                DEVICE,

                non_blocking=True
            )

            # Synchronize CUDA before timing

            if DEVICE.type == "cuda":

                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model(images)

            if DEVICE.type == "cuda":

                torch.cuda.synchronize()

            end_time = time.perf_counter()

            total_inference_time += (

                end_time - start_time
            )

            total_images += (

                images.size(0)
            )

            probabilities = torch.softmax(

                outputs,

                dim=1
            )

            predictions = torch.argmax(

                probabilities,

                dim=1
            )

            all_labels.extend(

                labels.cpu().numpy()
            )

            all_predictions.extend(

                predictions.cpu().numpy()
            )

            all_probabilities.extend(

                probabilities.cpu().numpy()
            )

    # ------------------------------------------------------
    # CONVERT TO NUMPY
    # ------------------------------------------------------

    all_labels = np.array(

        all_labels
    )

    all_predictions = np.array(

        all_predictions
    )

    all_probabilities = np.array(

        all_probabilities
    )

        # ==========================================================
    # CALCULATE METRICS
    # ==========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    # ----------------------------------------------------------
    # ROC-AUC
    # ----------------------------------------------------------

    try:

        roc_auc = roc_auc_score(

            all_labels,

            all_probabilities,

            multi_class="ovr",

            average="macro"
        )

    except ValueError as error:

        print(
            "\nROC-AUC could not be calculated:"
        )

        print(error)

        roc_auc = 0.0

    # ----------------------------------------------------------
    # PR-AUC
    # ----------------------------------------------------------

    try:

        one_hot_labels = np.eye(
            NUM_CLASSES
        )[all_labels]

        pr_auc = average_precision_score(

            one_hot_labels,

            all_probabilities,

            average="macro"
        )

    except ValueError as error:

        print(
            "\nPR-AUC could not be calculated:"
        )

        print(error)

        pr_auc = 0.0

    # ----------------------------------------------------------
    # INFERENCE TIME
    # ----------------------------------------------------------

    inference_time_ms = (

        total_inference_time

        / total_images

    ) * 1000

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================

    print("\n" + "=" * 65)

    print(
        "FINAL TEST RESULTS:",
        display_name
    )

    print("=" * 65)

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Macro Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Macro F1-score : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{pr_auc:.4f}"
    )

    print(
        f"Parameters     : "
        f"{total_parameters / 1_000_000:.2f} M"
    )

    print(
        f"Inference Time : "
        f"{inference_time_ms:.3f} ms/image"
    )

    print("=" * 65)

    # ==========================================================
    # CLASSIFICATION REPORT
    # ==========================================================

    report = classification_report(

        all_labels,

        all_predictions,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0
    )

    print(
        "\nClassification Report:\n"
    )

    print(report)

    # ==========================================================
    # SAVE METRICS
    # ==========================================================

    file_name = (

        model_name

        + "_metrics.json"
    )

    metrics_path = os.path.join(

        "outputs/metrics",

        file_name
    )

    metrics = {

        "model": display_name,

        "model_name": model_name,

        "accuracy": float(
            accuracy * 100
        ),

        "precision": float(
            precision * 100
        ),

        "recall": float(
            recall * 100
        ),

        "f1_score": float(
            f1 * 100
        ),

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "parameters": int(
            total_parameters
        ),

        "parameters_million": float(

            total_parameters

            / 1_000_000
        ),

        "inference_time_ms": float(
            inference_time_ms
        ),

        "test_images": int(
            total_images
        )
    }

    with open(

        metrics_path,

        "w"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4
        )

    print(

        "\nMetrics saved:",

        metrics_path
    )

    # ==========================================================
    # SAVE CLASSIFICATION REPORT
    # ==========================================================

    report_file = (

        model_name

        + "_classification_report.txt"
    )

    report_path = os.path.join(

        "outputs/evaluation_reports",

        report_file
    )

    with open(

        report_path,

        "w"

    ) as file:

        file.write(

            f"MODEL: {display_name}\n"

        )

        file.write(

            "=" * 65

        )

        file.write(

            "\n\n"

        )

        file.write(

            report
        )

    print(

        "Classification report saved:",

        report_path
    )

    # ==========================================================
    # RETURN RESULTS
    # ==========================================================

    return metrics


# ==========================================================
# EVALUATE ALL MODELS
# ==========================================================

all_results = []

for display_name, config in MODELS.items():

    result = evaluate_model(

        display_name=display_name,

        model_name=config["model_name"],

        checkpoint_path=config["checkpoint"],

        image_size=config["image_size"]
    )

    if result is not None:

        all_results.append(
            result
        )


# ==========================================================
# SAVE COMBINED RESULTS
# ==========================================================

if len(all_results) > 0:

    comparison_dataframe = pd.DataFrame(

        all_results
    )

    comparison_columns = [

        "model",

        "accuracy",

        "precision",

        "recall",

        "f1_score",

        "roc_auc",

        "pr_auc",

        "parameters_million",

        "inference_time_ms",

        "test_images"
    ]

    comparison_dataframe = (

        comparison_dataframe[

            comparison_columns

        ]
    )

    comparison_path = (

        "outputs/metrics/"
        "all_models_comparison.csv"
    )

    comparison_dataframe.to_csv(

        comparison_path,

        index=False
    )

    print("\n" + "=" * 70)

    print(
        "ALL MODEL EVALUATIONS COMPLETED"
    )

    print("=" * 70)

    print(

        comparison_dataframe.to_string(

            index=False
        )
    )

    print(

        "\nCombined comparison saved:",

        comparison_path
    )

else:

    print(

        "\nNo models were evaluated."
    )