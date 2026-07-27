"""Algorithm 1: Semantic-Anchor Guided Global Learning.

Implements Stage 1 of the benchmark exactly as described by the global
semantic-anchor guided training protocol. The verified model architecture,
optimizer and scheduler, losses, randomization, client partitioning, client
order, communication loop, FedAvg/FedAvgM updates, and evaluation procedure
are preserved.

Set dataset, CLIP, description, and output locations in the configuration
section below, or override them with the corresponding environment variables.
"""

import copy
import gc
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

from transformers import CLIPModel, CLIPProcessor


# Configuration

# Dataset
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
TRAIN_DIR = os.environ.get("TRAIN_DIR", "blc28_train")
TEST_DIR = os.environ.get("TEST_DIR", "blc28_test")
TARGET_SIZE = (224, 224)

# Training
BATCH_SIZE = 32
NUM_ROUNDS = 500
LOCAL_EPOCHS = 5
NUM_CLIENTS = 28
NUM_CLASSES = 28
SEED = 43
SHUFFLE_CLASSES = False

# Knowledge Distillation
USE_DESC_TEACHER = True
DESCRIPTION_CSV = Path(
    os.environ.get(
        "DESCRIPTION_CSV",
        "metadata/human_descriptions.csv",
    )
)
CLIP_CHECKPOINT = Path(
    os.environ.get(
        "CLIP_CHECKPOINT",
        "checkpoints/clip_teacher",
    )
)
KD_LAMBDA = 0.1
KD_TEMP = 4.0
KD_WARMUP_ROUNDS = 1
KD_EVERY = 1

# Federated Learning
CLIENT_SPLIT = "label_skew"
DIRICHLET_ALPHA = 10
FEDPROX_MU = 2e-3
FEDAVGM_BETA = 0.9

# Output
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
METRICS_DIR = OUTPUT_DIR / "metrics"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# Automatic CUDA selection is used for portable benchmark execution.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def configure_logging():
    """Configure concise console and file logging for the benchmark run."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS_DIR / "stage1.log")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )


LOGGER = logging.getLogger(__name__)


def kd_lambda_schedule(
    round_idx,
    warmup_rounds=KD_WARMUP_ROUNDS,
    ramp_end=50,
    max_lambda=0.8,
):
    """Return the verified round-dependent KD coefficient."""
    round_number = round_idx + 1
    if round_number <= warmup_rounds:
        return 0.0
    if round_number >= ramp_end:
        return float(max_lambda)
    progress = (round_number - warmup_rounds) / float(
        ramp_end - warmup_rounds
    )
    return float(max_lambda) * progress


def load_human_descriptions(csv_path):
    """Load class descriptions from the benchmark metadata CSV."""
    dataframe = pd.read_csv(csv_path)
    descriptions = {}
    for _, row in dataframe.iterrows():
        class_name = str(row["class_name"]).strip()
        description = str(row["descriptions"])
        parts = []
        for chunk in description.replace("\r", "\n").split("\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts.append(chunk)

        if len(parts) == 1 and "," in parts[0]:
            parts = [
                part.strip()
                for part in parts[0].split(",")
                if part.strip()
            ]

        descriptions[class_name] = parts
    return descriptions


def load_clip_teacher(device):
    """Load the verified fine-tuned CLIP teacher checkpoint."""
    clip_model = CLIPModel.from_pretrained(
        str(CLIP_CHECKPOINT)
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained(
        str(CLIP_CHECKPOINT)
    )

    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    for parameter in clip_model.text_projection.parameters():
        parameter.requires_grad = True

    for parameter in clip_model.visual_projection.parameters():
        parameter.requires_grad = True

    return clip_model, clip_processor


def freeze_bn_only(model):
    """Freeze batch-normalization parameters and running statistics."""
    for module in model.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


@torch.no_grad()
def build_class_text_bank(
    categories,
    desc_map,
    clip_model,
    clip_processor,
    device,
    max_desc_per_class=32,
):
    """Build the verified prompt list, class mapping, and CLIP embeddings."""
    texts, text_class = [], []
    missing = []

    for class_id, class_name in enumerate(categories):
        if class_name not in desc_map:
            missing.append(class_name)
            continue

        lines = desc_map[class_name][:max_desc_per_class]
        lines = [
            line.strip(" ,")
            for line in lines
            if len(line.strip()) > 2
        ]

        prompts = [
            (
                f"a microscopy image of a {class_name} cell in a "
                f"peripheral blood smear (Wright stain), {line}"
            )
            for line in lines
        ]

        anchors = [
            (
                f"a microscopy image of a {class_name} cell in a "
                "peripheral blood smear (Wright stain)"
            ),
            (
                f"peripheral blood smear showing a {class_name} cell "
                "(Wright stain)"
            ),
            (
                f"a {class_name} cell under the microscope "
                "(hematology)"
            ),
        ]
        prompts = anchors + prompts

        for prompt in prompts:
            texts.append(prompt)
            text_class.append(class_id)

    if missing:
        print(
            "[WARN] Missing descriptions for these categories "
            "(folder names):"
        )
        for class_name in missing:
            print("  -", class_name)

    tokenized = clip_processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    tokenized = {
        key: value.to(device)
        for key, value in tokenized.items()
    }

    text_features = clip_model.get_text_features(**tokenized)
    text_features = text_features / (
        text_features.norm(dim=-1, keepdim=True) + 1e-12
    )
    return texts, text_class, text_features


@torch.no_grad()
def teacher_logits_from_clip(
    clip_model,
    clip_processor,
    device,
    batch_images_tensor,
    text_embeds,
    text_class,
    num_classes,
    temperature=2.0,
):
    """Return verified class-aggregated CLIP logits for one image batch."""
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=batch_images_tensor.device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=batch_images_tensor.device,
    ).view(1, 3, 1, 1)
    images = (batch_images_tensor * std + mean).clamp(0, 1)

    pil_images = []
    for image_index in range(images.size(0)):
        image_array = (
            images[image_index]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)
        pil_images.append(Image.fromarray(image_array))

    inputs = clip_processor(
        images=pil_images,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / (
        image_features.norm(dim=-1, keepdim=True) + 1e-12
    )

    similarities = image_features @ text_embeds.t()

    logits = torch.full(
        (images.size(0), num_classes),
        -1e9,
        device=device,
    )
    class_columns = defaultdict(list)
    for column_index, class_id in enumerate(text_class):
        class_columns[class_id].append(column_index)

    for class_id, columns in class_columns.items():
        logits[:, class_id] = similarities[:, columns].mean(dim=1)

    return logits / float(temperature)


np.random.seed(SEED)
torch.manual_seed(SEED)


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


# Dataset


def create_dataset(path):
    """Index supported benchmark images and their global labels."""
    records = []
    for category in categories:
        category_path = os.path.join(path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                records.append(
                    [file_path, label_map[category]]
                )
    return pd.DataFrame(
        records,
        columns=["file_path", "label"],
    )


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
            for client_index, size in enumerate(
                client_sample_sizes
            ):
                end_index = start_index + size
                if size > 0:
                    client_dataframes[client_index] = pd.concat(
                        [
                            client_dataframes[client_index],
                            label_group.iloc[
                                start_index:end_index
                            ],
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
    total_per_class = (
        dataframe["label"].value_counts().sort_index()
    )

    return client_dataframes


# Model


def get_mobilenetv3_model(num_classes, pretrained=True):
    """Create the verified pretrained MobileNetV3-Large classifier."""
    weights = (
        MobileNet_V3_Large_Weights.DEFAULT
        if pretrained
        else None
    )
    model = models.mobilenet_v3_large(weights=weights)

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes,
    )

    return model


# Training


def train_model(
    model,
    train_loader,
    epochs,
    device,
    clip_model=None,
    clip_processor=None,
    text_embeds=None,
    text_class=None,
    kd_lambda=0.0,
    kd_temp=2.0,
    kd_every=1,
    global_params=None,
    prox_mu=0.0,
):
    """Run the verified local optimizer, scheduler, KD, and FedProx steps."""
    model.to(device)
    model.train()
    freeze_bn_only(model)

    parameter_groups = []

    parameter_groups.append(
        {
            "params": model.classifier.parameters(),
            "lr": 5e-4,
        }
    )

    num_late_blocks = 3
    late_blocks = list(
        model.features.children()
    )[-num_late_blocks:]
    late_parameters = []
    for block in late_blocks:
        late_parameters += list(block.parameters())

    parameter_groups.append(
        {
            "params": late_parameters,
            "lr": 2e-4,
        }
    )

    early_blocks = list(
        model.features.children()
    )[:-num_late_blocks]
    early_parameters = []
    for block in early_blocks:
        early_parameters += list(block.parameters())

    parameter_groups.append(
        {
            "params": early_parameters,
            "lr": 5e-5,
        }
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    step = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * epochs,
    )

    for epoch in range(epochs):
        for images, labels in train_loader:
            step += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(outputs, tuple):
                student_logits, auxiliary_logits = outputs
            else:
                student_logits = outputs
                auxiliary_logits = None

            loss = criterion(student_logits, labels)

            if auxiliary_logits is not None:
                loss = (
                    loss
                    + 0.3
                    * criterion(auxiliary_logits, labels)
                )

            use_kd = (
                clip_model is not None
                and clip_processor is not None
                and text_embeds is not None
                and text_class is not None
                and (step % kd_every == 0)
            )

            if use_kd:
                with torch.no_grad():
                    teacher_logits = teacher_logits_from_clip(
                        clip_model,
                        clip_processor,
                        device,
                        batch_images_tensor=images,
                        text_embeds=text_embeds,
                        text_class=text_class,
                        num_classes=num_classes,
                        temperature=kd_temp,
                    )

                    teacher_probabilities = torch.softmax(
                        teacher_logits,
                        dim=1,
                    )
                    confidence = teacher_probabilities.max(
                        dim=1
                    ).values
                    confidence_weight = (
                        (
                            confidence - 1.0 / num_classes
                        )
                        / (
                            1.0 - 1.0 / num_classes
                        )
                    ).clamp(0, 1)

                student_log_probabilities = torch.log_softmax(
                    student_logits / kd_temp,
                    dim=1,
                )
                kd_loss_per_sample = torch.sum(
                    teacher_probabilities
                    * (
                        torch.log(
                            teacher_probabilities + 1e-12
                        )
                        - student_log_probabilities
                    ),
                    dim=1,
                )
                kd_loss = (
                    confidence_weight * kd_loss_per_sample
                ).mean()
                loss = (
                    loss
                    + kd_lambda
                    * (kd_temp**2)
                    * kd_loss
                )

            if prox_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        proximal_term = (
                            proximal_term
                            + torch.sum(
                                (
                                    parameter
                                    - global_params[name]
                                )
                                ** 2
                            )
                        )
                loss = loss + 0.5 * prox_mu * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()


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


# Federated aggregation


def fedavg(models, weights):
    """Return the verified sample-weighted FedAvg model state."""
    if len(models) == 0:
        return {}
    if len(models) == 1:
        return copy.deepcopy(models[0].state_dict())

    normalized_weights = np.asarray(
        weights,
        dtype=np.float64,
    )
    normalized_weights = normalized_weights / (
        normalized_weights.sum() + 1e-12
    )

    state_dicts = [model.state_dict() for model in models]
    averaged_state = {}
    for key in state_dicts[0].keys():
        averaged_state[key] = sum(
            float(weight) * state_dict[key]
            for weight, state_dict in zip(
                normalized_weights,
                state_dicts,
            )
        )
    return averaged_state


def fedavgm_update(
    global_state,
    avg_state,
    velocity,
    beta=0.9,
):
    """Apply the verified FedAvgM server update and return its velocity."""
    if velocity is None:
        velocity = {
            key: torch.zeros_like(value)
            for key, value in global_state.items()
        }
    new_state = {}
    for key in global_state.keys():
        delta = avg_state[key] - global_state[key]
        velocity[key] = beta * velocity[key] + delta
        new_state[key] = global_state[key] + velocity[key]
    return new_state, velocity


# Main federated loop


def fedavg_main(num_rounds, local_epochs):
    """Run the verified Stage 1 federated benchmark protocol."""
    global_accuracy_history = []
    print(f"Using device: {DEVICE}")

    train_dataframe = create_dataset(
        DATA_ROOT / TRAIN_DIR
    )
    test_dataframe = create_dataset(
        DATA_ROOT / TEST_DIR
    )
    client_dataframes = get_client_data_splits(
        train_dataframe,
        NUM_CLIENTS,
        CLIENT_SPLIT,
        DIRICHLET_ALPHA,
    )

    if USE_DESC_TEACHER:
        description_map = load_human_descriptions(
            DESCRIPTION_CSV
        )
        clip_model, clip_processor = load_clip_teacher(DEVICE)

        (
            teacher_texts,
            teacher_text_class,
            teacher_text_embeddings,
        ) = build_class_text_bank(
            categories=categories,
            desc_map=description_map,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=DEVICE,
            max_desc_per_class=32,
        )
    else:
        clip_model = clip_processor = None
        teacher_text_embeddings = teacher_text_class = None

    print(
        "\nClient x Class sample-count matrix "
        "(rows=clients, columns=classes)"
    )

    class_labels = sorted(
        train_dataframe["label"].unique()
    )

    client_class_matrix = pd.DataFrame(
        0,
        index=[
            f"Client {client_id}"
            for client_id in range(NUM_CLIENTS)
        ],
        columns=class_labels,
    )

    for client_id, client_dataframe in enumerate(
        client_dataframes
    ):
        class_counts = (
            client_dataframe["label"].value_counts()
        )
        for class_label, count in class_counts.items():
            client_class_matrix.loc[
                f"Client {client_id}",
                class_label,
            ] = count

    print(client_class_matrix)

    print(
        "\nClient x Class matrix "
        "(% of total samples per class)"
    )
    total_per_class = (
        train_dataframe["label"]
        .value_counts()
        .sort_index()
    )
    client_class_matrix_percent = client_class_matrix.copy()
    for class_label in class_labels:
        total = total_per_class[class_label]
        client_class_matrix_percent[class_label] = (
            (
                client_class_matrix[class_label]
                / total
            )
            * 100
        ).round(2) if total > 0 else 0.0

    print(client_class_matrix_percent)

    print("========================================================\n")

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

    global_model = get_mobilenetv3_model(
        num_classes=num_classes
    ).to(DEVICE)
    global_weights = global_model.state_dict()

    for round_index in range(num_rounds):
        print(f"\n--- Global Round {round_index + 1} ---")
        round_start = time.time()
        LOGGER.info(
            "Starting communication round %d/%d",
            round_index + 1,
            num_rounds,
        )

        kd_lambda_round = kd_lambda_schedule(
            round_index,
            warmup_rounds=KD_WARMUP_ROUNDS,
            ramp_end=50,
            max_lambda=0.8,
        )
        print(
            f" KD_LAMBDA (round) = "
            f"{kd_lambda_round:.4f}"
        )

        client_models = []
        client_sizes = []
        server_velocity = None

        count = 0
        for client_id in range(NUM_CLIENTS):
            print(
                f" Training Client {count + 1}",
                "id : ",
                client_id + 1,
            )
            client_start = time.time()

            local_model = get_mobilenetv3_model(
                num_classes=num_classes
            ).to(DEVICE)
            local_model.load_state_dict(global_weights)
            global_parameters = {
                key: value.detach().clone().to(DEVICE)
                for key, value in global_weights.items()
            }

            use_kd_this_round = (
                round_index + 1
            ) > KD_WARMUP_ROUNDS

            train_model(
                local_model,
                client_loaders[client_id],
                epochs=local_epochs,
                device=DEVICE,
                clip_model=(
                    clip_model
                    if (
                        USE_DESC_TEACHER
                        and use_kd_this_round
                    )
                    else None
                ),
                clip_processor=(
                    clip_processor
                    if (
                        USE_DESC_TEACHER
                        and use_kd_this_round
                    )
                    else None
                ),
                text_embeds=(
                    teacher_text_embeddings
                    if (
                        USE_DESC_TEACHER
                        and use_kd_this_round
                    )
                    else None
                ),
                text_class=(
                    teacher_text_class
                    if (
                        USE_DESC_TEACHER
                        and use_kd_this_round
                    )
                    else None
                ),
                kd_lambda=kd_lambda_round,
                kd_temp=KD_TEMP,
                kd_every=KD_EVERY,
                global_params=global_parameters,
                prox_mu=FEDPROX_MU,
            )

            client_models.append(local_model)
            client_sizes.append(
                len(client_loaders[client_id].dataset)
            )

            client_time = time.time() - client_start
            LOGGER.info(
                "Client %d training time: %.2f seconds",
                client_id + 1,
                client_time,
            )
            count += 1

        averaged_state = fedavg(
            client_models,
            weights=client_sizes,
        )
        global_weights, server_velocity = fedavgm_update(
            global_weights,
            averaged_state,
            server_velocity,
            beta=FEDAVGM_BETA,
        )
        global_model.load_state_dict(global_weights)

        accuracy = evaluate_model(
            global_model,
            test_loader,
            device=DEVICE,
        )
        print(
            f" Test Accuracy after Round "
            f"{round_index + 1}: {accuracy:.2f}%"
        )
        global_accuracy_history.append(accuracy)

        # This file stores the same ordered global-accuracy history as the
        # verified implementation. Only its release-facing path and concise
        # filename differ.
        np.save(
            METRICS_DIR / "global_accuracy.npy",
            global_accuracy_history,
        )
        LOGGER.info(
            "Saved global accuracy metrics to %s",
            METRICS_DIR,
        )

        round_time = time.time() - round_start
        print(
            f" Round {round_index + 1} total time: "
            f"{round_time:.2f} seconds"
        )

        del client_models
        torch.cuda.empty_cache()
        gc.collect()

    print(
        f"\nFinal FedAvg Accuracy after "
        f"{num_rounds} rounds: {accuracy:.2f}%"
    )


# Reproducibility protocol:
#   Random seed: SEED
#   Dataset: DATA_ROOT / TRAIN_DIR and DATA_ROOT / TEST_DIR
#   Benchmark protocol: CLIENT_SPLIT with semantic-anchor supervision
#   Communication rounds: NUM_ROUNDS
if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()
    LOGGER.info(
        "Stage 1 | dataset=%s | clients=%d | split=%s | "
        "rounds=%d | local epochs=%d | seed=%d | device=%s",
        DATA_ROOT,
        NUM_CLIENTS,
        CLIENT_SPLIT,
        NUM_ROUNDS,
        LOCAL_EPOCHS,
        SEED,
        DEVICE,
    )
    fedavg_main(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
    )
