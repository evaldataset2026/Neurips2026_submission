"""Algorithm 2: Client Personalization.

Implements Stage 2 of the benchmark exactly as described by the client
personalization protocol. Each client starts from the verified global model,
replaces its classifier with a client-label classifier, freezes the backbone
except for the final two feature blocks, and performs the verified local
fine-tuning procedure.

Set dataset, checkpoint, and output locations in the configuration section
below, or override them with the corresponding environment variables.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights


# Configuration

# Dataset
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
TRAIN_DIR = os.environ.get("TRAIN_DIR", "18class_train")
TEST_DIR = os.environ.get("TEST_DIR", "18class_test")

# Training
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_ROUNDS = 500
LOCAL_EPOCHS = 5
NUM_CLIENTS = 3
NUM_CLASSES = 18
SEED = 43
SHUFFLE_CLASSES = False
PERSONALIZATION_EPOCHS = 50

# Knowledge Distillation
USE_DESC_TEACHER = True
DESCRIPTION_CSV = Path(
    os.environ.get(
        "DESCRIPTION_CSV",
        "metadata/human_descriptions.csv",
    )
)
KD_LAMBDA = 0.1
KD_TEMP = 4.0
KD_WARMUP_ROUNDS = 1
KD_EVERY = 1

# Federated Learning
CLIENT_SPLIT = "label_skew"
DIRICHLET_ALPHA = 10

# Checkpoint and output
GLOBAL_CHECKPOINT = Path(
    os.environ.get("GLOBAL_CHECKPOINT", "global_model_mlc_3.pth")
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
METRICS_DIR = OUTPUT_DIR / "metrics"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# Automatic CUDA selection is used for portable benchmark execution.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


np.random.seed(SEED)
torch.manual_seed(SEED)


# Dataset


train_path = DATA_ROOT / TRAIN_DIR
categories = sorted(
    [
        directory
        for directory in os.listdir(train_path)
        if os.path.isdir(train_path / directory)
    ]
)
label_map = {
    category: index for index, category in enumerate(categories)
}
num_classes = len(categories)


def create_dataset(path):
    """Index supported images under the configured class directories."""
    records = []
    for category in categories:
        category_path = os.path.join(path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff")
            ):
                records.append([file_path, label_map[category]])
    return pd.DataFrame(records, columns=["file_path", "label"])


class ImageDataset(Dataset):
    """Image dataset backed by a dataframe of paths and global labels."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index, 0]
        label = self.dataframe.iloc[index, 1]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


train_transform = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
        ),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# Client partitioning

# Benchmark partition protocols:
#
# EC-DLS ("label_skew") assigns disjoint class sets to clients. It evaluates
# personalization when each client observes a distinct label subspace.
#
# Dirichlet ("dirichlet") allocates examples from each class across clients
# according to a Dirichlet draw. It evaluates heterogeneous non-IID class
# proportions while retaining potential label overlap between clients.
#
# Both branches below preserve the verified benchmark partition procedures.
def get_client_data_splits(
    dataframe,
    num_clients,
    split_type,
    alpha,
):
    """Return one verified EC-DLS or Dirichlet dataframe per client."""
    client_dataframes = [
        pd.DataFrame(columns=dataframe.columns)
        for _ in range(num_clients)
    ]

    if split_type == "dirichlet":
        unique_labels = sorted(dataframe["label"].unique())
        label_distributions = np.random.dirichlet(
            [alpha] * num_clients,
            size=len(unique_labels),
        )
        label_groups = [
            dataframe[dataframe["label"] == label].copy()
            for label in unique_labels
        ]
        client_dataframes = [
            pd.DataFrame() for _ in range(num_clients)
        ]

        for label_index, label_group in enumerate(label_groups):
            label_group = label_group.sample(
                frac=1,
                random_state=SEED + label_index,
            )
            client_sample_sizes = (
                label_distributions[label_index] * len(label_group)
            ).astype(int)
            remaining = len(label_group) - sum(client_sample_sizes)
            if remaining > 0:
                indices = np.argsort(
                    -label_distributions[label_index]
                )[:remaining]
                for index in indices:
                    client_sample_sizes[index] += 1
            start_index = 0
            for client_index, size in enumerate(client_sample_sizes):
                end_index = start_index + size
                if size > 0:
                    client_dataframes[client_index] = pd.concat(
                        [
                            client_dataframes[client_index],
                            label_group.iloc[start_index:end_index],
                        ]
                    )
                start_index = end_index

    elif split_type == "label_skew":
        class_labels = sorted(dataframe["label"].unique())
        try:
            total_classes = int(num_classes)
        except Exception:
            total_classes = len(class_labels)

        random_generator = random.Random(SEED)
        class_order = list(range(total_classes))

        classes_per_client = total_classes // num_clients
        client_class_assignments = []
        start = 0
        for client_id in range(num_clients):
            if client_id == num_clients - 1:
                assigned_classes = class_order[start:]
            else:
                assigned_classes = class_order[
                    start : start + classes_per_client
                ]
            client_class_assignments.append(assigned_classes)
            start += classes_per_client

        for client_id, assigned_classes in enumerate(
            client_class_assignments
        ):
            if len(assigned_classes) == 0:
                client_dataframes[client_id] = pd.DataFrame(
                    columns=dataframe.columns
                )
                continue
            client_dataframe = dataframe[
                dataframe["label"].isin(assigned_classes)
            ].copy()
            if len(client_dataframe) > 0:
                client_dataframe = client_dataframe.sample(
                    frac=1,
                    random_state=SEED + client_id,
                ).reset_index(drop=True)
            client_dataframes[client_id] = client_dataframe

        print(
            "\nLabel-skew class assignment "
            "(client_id : assigned_class_ids):"
        )
        for client_id, assigned_classes in enumerate(
            client_class_assignments
        ):
            print(
                f" Client {client_id}: classes {assigned_classes} "
                f"-> {len(client_dataframes[client_id])} samples"
            )

    else:
        raise ValueError(f"Unknown split_type '{split_type}'")

    class_labels = sorted(dataframe["label"].unique())
    num_clients_returned = len(client_dataframes)
    total_per_class = dataframe["label"].value_counts().sort_index()

    return client_dataframes


# Model


def get_mobilenetv3_model(num_classes, pretrained=True):
    """Create the verified MobileNetV3-Large global classifier."""
    weights = (
        MobileNet_V3_Large_Weights.DEFAULT
        if pretrained
        else None
    )
    model = models.mobilenet_v3_large(weights=weights)

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)

    return model


# Evaluation


def evaluate_model(model, loader, device):
    """Return top-1 classification accuracy as a percentage."""
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0


# Main


def main():
    """Run the verified Stage 2 client-personalization protocol."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Stage 2 | dataset=%s | clients=%d | split=%s | "
        "personalization epochs=%d | seed=%d | device=%s",
        DATA_ROOT,
        NUM_CLIENTS,
        CLIENT_SPLIT,
        PERSONALIZATION_EPOCHS,
        SEED,
        DEVICE,
    )

    train_dataframe = create_dataset(DATA_ROOT / TRAIN_DIR)
    test_dataframe = create_dataset(DATA_ROOT / TEST_DIR)
    client_dataframes = get_client_data_splits(
        train_dataframe,
        NUM_CLIENTS,
        CLIENT_SPLIT,
        DIRICHLET_ALPHA,
    )

    client_test_dataframes = []

    print(
        "\n================ CLIENT TEST SPLITS "
        "(FROM GLOBAL TEST SET) ================="
    )

    for client_id in range(NUM_CLIENTS):
        client_classes = sorted(
            client_dataframes[client_id]["label"].unique()
        )

        client_test_dataframe = test_dataframe[
            test_dataframe["label"].isin(client_classes)
        ].copy()

        client_test_dataframe = client_test_dataframe.sample(
            frac=1,
            random_state=SEED + client_id,
        ).reset_index(drop=True)

        client_test_dataframes.append(client_test_dataframe)

        print(f"\nClient {client_id}:")
        print(f"  Classes: {client_classes}")
        print(f"  Test samples: {len(client_test_dataframe)}")

        print("  Class distribution:")
        print(
            client_test_dataframe["label"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print(
        "=========================================================="
        "===============\n"
    )

    client_test_loaders = []

    for client_id in range(NUM_CLIENTS):
        loader = DataLoader(
            ImageDataset(
                client_test_dataframes[client_id],
                transform=test_transform,
            ),
            batch_size=256,
            shuffle=False,
        )
        client_test_loaders.append(loader)

    client_loaders = []
    for client_dataframe in client_dataframes:
        train_dataset = ImageDataset(
            client_dataframe,
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        client_loaders.append(train_loader)

    test_dataset = ImageDataset(
        test_dataframe,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # The global checkpoint stores ``model_state_dict``, the verified Stage 1
    # model parameters used to initialize every personalization client. Loading
    # the same state for every client ensures a common global starting point.
    checkpoint = torch.load(
        GLOBAL_CHECKPOINT,
        map_location=DEVICE,
    )

    model = get_mobilenetv3_model(
        num_classes=num_classes
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoaded model")
    LOGGER.info("Loaded global checkpoint from %s", GLOBAL_CHECKPOINT)

    accuracy = evaluate_model(
        model,
        test_loader,
        device=DEVICE,
    )

    print(f" Global Test Accuracy (reloaded): {accuracy:.2f}%")

    for client_id in range(NUM_CLIENTS):
        accuracy = evaluate_model(
            model,
            client_test_loaders[client_id],
            device=DEVICE,
        )

        print(
            f" Local Test Accuracy (reloaded) of {client_id} "
            f"---- {accuracy:.2f}%"
        )

    print("\n================ STAGE 2: PERSONALIZATION =================")

    personalized_models = []
    personalized_accuracy_history = []

    for client_id in range(NUM_CLIENTS):
        print(f"\n--- Client {client_id} Personalization ---")
        LOGGER.info(
            "Personalizing client %d/%d",
            client_id + 1,
            NUM_CLIENTS,
        )

        local_model = get_mobilenetv3_model(
            num_classes=num_classes
        ).to(DEVICE)
        local_model.load_state_dict(checkpoint["model_state_dict"])

        client_classes = sorted(
            client_dataframes[client_id]["label"].unique()
        )
        num_local_classes = len(client_classes)

        input_features = local_model.classifier[-1].in_features
        local_model.classifier[-1] = nn.Linear(
            input_features,
            num_local_classes,
        ).to(DEVICE)

        class_map = {
            class_label: index
            for index, class_label in enumerate(client_classes)
        }

        class RemappedDataset(Dataset):
            """Map global labels to this client's contiguous label space."""

            def __init__(self, dataframe, transform):
                self.dataframe = dataframe.reset_index(drop=True)
                self.transform = transform

            def __len__(self):
                return len(self.dataframe)

            def __getitem__(self, index):
                image_path = self.dataframe.iloc[index]["file_path"]
                label = self.dataframe.iloc[index]["label"]
                image = Image.open(image_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image, class_map[label]

        train_loader = DataLoader(
            RemappedDataset(
                client_dataframes[client_id],
                train_transform,
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        test_loader = DataLoader(
            RemappedDataset(
                client_test_dataframes[client_id],
                test_transform,
            ),
            batch_size=256,
        )

        for name, parameter in local_model.features.named_parameters():
            parameter.requires_grad = False

        for layer in list(
            local_model.features.children()
        )[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(
                lambda parameter: parameter.requires_grad,
                local_model.parameters(),
            ),
            lr=1e-3,
        )

        criterion = nn.CrossEntropyLoss()

        local_model.train()

        for epoch in range(PERSONALIZATION_EPOCHS):
            print(f"--- Epoch {epoch} ---")

            for images, labels in train_loader:
                images, labels = (
                    images.to(DEVICE),
                    labels.to(DEVICE),
                )

                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        local_model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = (
                    images.to(DEVICE),
                    labels.to(DEVICE),
                )
                predictions = local_model(images).argmax(1)
                correct += predictions.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(
            f" Client {client_id} Personalized Accuracy: "
            f"{accuracy:.2f}%"
        )
        personalized_accuracy_history.append(accuracy)

        personalized_models.append(local_model)

    # This file stores the same ordered per-client accuracy array as the
    # verified implementation; only its release-facing directory and concise
    # filename differ.
    np.save(
        METRICS_DIR / "stage2_local_accuracy.npy",
        np.array(personalized_accuracy_history),
    )
    LOGGER.info("Saved Stage 2 metrics to %s", METRICS_DIR)


# Reproducibility protocol:
#   Random seed: SEED
#   Dataset: DATA_ROOT / TRAIN_DIR and DATA_ROOT / TEST_DIR
#   Benchmark protocol: CLIENT_SPLIT client personalization
#   Communication rounds: Stage 2 loads the completed global checkpoint
if __name__ == "__main__":
    main()
