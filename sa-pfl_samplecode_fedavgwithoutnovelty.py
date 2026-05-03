import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights
import time
import random

# ================== CONFIG ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 28
NUM_CLASSES = 28
ROUNDS = 500
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
DIRICHLET_ALPHA = 0.5
SEED = 43
CLIENT_SPLIT = "label_skew"   # options: "dirichlet" or "label_skew"

base_directory = './data'
train_dir, test_dir = 'blc28_train', 'blc28_test'
target_size = (224,224)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ================== DATA ==================
train_path = os.path.join(base_directory, train_dir)
test_path = os.path.join(base_directory, test_dir)

VALID_EXTS = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')

categories = sorted([d for d in os.listdir(train_path)
                     if os.path.isdir(os.path.join(train_path, d))])

label_map = {category: idx for idx, category in enumerate(categories)}

# ================== TRANSFORMS ==================
train_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================== DATASET ==================
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['file_path']
        label = self.df.iloc[idx]['label']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataset(path):
    data = []
    for category in categories:
        category_path = os.path.join(path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):  
                data.append([file_path, label_map[category]])
    return pd.DataFrame(data, columns=['file_path', 'label'])

# ================== DIRICHLET SPLIT ==================
def dirichlet_split(df, num_clients, alpha):
    client_dfs = [pd.DataFrame(columns=df.columns) for _ in range(num_clients)]
    labels = df['label'].unique()

    for label in labels:
        label_df = df[df['label']==label]
        proportions = np.random.dirichlet([alpha]*num_clients)
        sizes = (proportions * len(label_df)).astype(int)

        # fix rounding
        sizes[-1] += len(label_df) - sizes.sum()

        start = 0
        for i, size in enumerate(sizes):
            client_dfs[i] = pd.concat([
                client_dfs[i],
                label_df.iloc[start:start+size]
            ])
            start += size

    return [cdf.sample(frac=1).reset_index(drop=True) for cdf in client_dfs]

def label_skew_split(df, num_clients):
    """
    Fully disjoint class assignment across clients.
    Each client gets exclusive subset of classes.
    """
    client_dfs = []

    class_order = sorted(df['label'].unique())
    total_classes = len(class_order)

    base = total_classes // num_clients
    start = 0

    client_class_map = []

    for cid in range(num_clients):
        if cid == num_clients - 1:
            assigned = class_order[start:]
        else:
            assigned = class_order[start:start+base]

        client_df = df[df['label'].isin(assigned)].copy()
        client_df = client_df.sample(frac=1, random_state=SEED + cid).reset_index(drop=True)

        client_dfs.append(client_df)
        client_class_map.append(assigned)

        start += base

    print("\n================ LABEL SKEW ASSIGNMENT ================")
    for cid, classes in enumerate(client_class_map):
        print(f"Client {cid}: classes {classes} -> {len(client_dfs[cid])} samples")
    print("=====================================================\n")

    return client_dfs

# ================== MODEL ==================
class MobileNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = model.classifier[0].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x,1)

class PersonalHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, NUM_CLASSES)

    def forward(self, x):
        return self.fc(x)

# ================== TRAIN ==================
def train_client(backbone, head, loader):
    backbone.train()
    head.train()

    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()), lr=0.01
    )

    correct, total = 0,0

    for _ in range(LOCAL_EPOCHS):
        for x,y in loader:
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = head(backbone(x))
            loss = F.cross_entropy(out,y)
            loss.backward()
            optimizer.step()

            _,pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return correct/total if total > 0 else 0.0

# ================== TEST ==================
def evaluate(backbone, head, loader):
    backbone.eval()
    head.eval()

    correct, total = 0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = head(backbone(x))
            _,pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return correct/total

# ================== FEDAVG ==================
def fedavg(global_model, client_models, weights):
    global_dict = global_model.state_dict()
    w = np.array(weights)/sum(weights)

    for key in global_dict:
        global_dict[key] = sum(
            w[i]*client_models[i].state_dict()[key].float()
            for i in range(len(client_models))
        )

    global_model.load_state_dict(global_dict)

# ================== LOAD DATA ==================
df_train = create_dataset(train_path)
df_test  = create_dataset(test_path)
# ================== GLOBAL TEST STATS ==================

print("\n================ GLOBAL TEST SET =================")

global_test_dist = df_test['label'].value_counts().sort_index()

print(f"\nTotal global test samples: {len(df_test)}")

# print("====================================================\n")

if CLIENT_SPLIT == "dirichlet":
    client_dfs = dirichlet_split(df_train, NUM_CLIENTS, DIRICHLET_ALPHA)
elif CLIENT_SPLIT == "label_skew":
    client_dfs = label_skew_split(df_train, NUM_CLIENTS)
else:
    raise ValueError("Invalid CLIENT_SPLIT")# ================== DATA DISTRIBUTION STATS ==================

# print("\n================ DATA DISTRIBUTION =================")

class_labels = sorted(df_train['label'].unique())
num_clients_ret = len(client_dfs)

# 1. Total samples per class
total_per_class = df_train['label'].value_counts().sort_index()


# 2. Per-client class distribution
distribution_table = pd.DataFrame(
    0,
    index=[f"Client {i}" for i in range(num_clients_ret)],
    columns=class_labels
)

for i, client_df in enumerate(client_dfs):
    class_counts = client_df['label'].value_counts()
    for label, count in class_counts.items():
        distribution_table.loc[f"Client {i}", label] = count

print("\n📊 Per-client class distribution (counts):")
print(distribution_table)

# 3. Percentage version
# print("\n📊 Per-client class distribution (% of total per class):")
percent_table = distribution_table.copy()

for label in class_labels:
    total = total_per_class[label]
    percent_table[label] = (
        (distribution_table[label] / total) * 100
    ).round(2) if total > 0 else 0.0

# print("====================================================\n")

# ===== LOCAL TRAIN/VAL SPLIT =====
client_train_loaders = []
client_test_loaders = []
client_sizes = []

print("\n================ CLIENT LOCAL SPLITS =================")

for cid, cdf in enumerate(client_dfs):
    split = int(0.8 * len(cdf))

    train_df = cdf.iloc[:split]
    test_df  = cdf.iloc[split:]

    print(f"\nClient {cid}:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples : {len(test_df)}")

    # Class distribution in train
    train_dist = train_df['label'].value_counts().sort_index()

    # Class distribution in test
    test_dist = test_df['label'].value_counts().sort_index()

    train_loader = DataLoader(
        ImageDataset(train_df, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        ImageDataset(test_df, test_transform),
        batch_size=256
    )

    client_train_loaders.append(train_loader)
    client_test_loaders.append(test_loader)
    client_sizes.append(len(train_df))

print("====================================================\n")

# ===== GLOBAL TEST =====
global_test_loader = DataLoader(
    ImageDataset(df_test,test_transform),
    batch_size=256
)

# ================== INIT MODELS ==================
global_backbone = MobileNetBackbone().to(device)

client_backbones = [MobileNetBackbone().to(device) for _ in range(NUM_CLIENTS)]
client_heads = [PersonalHead(global_backbone.out_dim).to(device) for _ in range(NUM_CLIENTS)]

# ================== LOGGING ==================
client_local_acc = {i:[] for i in range(NUM_CLIENTS)}
global_acc = []

# ================== TRAINING ==================
for r in range(ROUNDS):
    print(f"\n===== ROUND {r+1} =====")

    client_models = []

    for i in range(NUM_CLIENTS):
        client_backbones[i].load_state_dict(global_backbone.state_dict())

        train_acc = train_client(
            client_backbones[i],
            client_heads[i],
            client_train_loaders[i]
        )

        local_test_acc = evaluate(
            client_backbones[i],
            client_heads[i],
            client_test_loaders[i]
        )

        client_local_acc[i].append(local_test_acc)

        print(f"Client {i} Train: {train_acc:.4f} | Local Test: {local_test_acc:.4f}")

        client_models.append(client_backbones[i])

    # ===== FEDAVG BACKBONE =====
    fedavg(global_backbone, client_models, client_sizes)

    # ===== GLOBAL MODEL (avg head) =====
    global_head = copy.deepcopy(client_heads[0])
    state_dicts = [h.state_dict() for h in client_heads]

    avg_state = {}
    weights = np.array(client_sizes) / sum(client_sizes)
    for k in state_dicts[0]:

        avg_state[k] = sum(
            weights[i] * state_dicts[i][k].float()
            for i in range(len(state_dicts))
        )

    global_head.load_state_dict(avg_state)
    global_head.to(device)

    # ===== GLOBAL TEST =====
    g_acc = evaluate(global_backbone, global_head, global_test_loader)
    global_acc.append(g_acc)

    print(f"GLOBAL TEST ACC: {g_acc:.4f}")

    # ===== SAVE =====
    np.save("client_local_acc_labelskew_nonovelty7.npy", np.array([client_local_acc[i] for i in range(NUM_CLIENTS)]))
    np.save("global_acc_labelskew_nonovelty7.npy", np.array(global_acc))

    del global_head
    torch.cuda.empty_cache()