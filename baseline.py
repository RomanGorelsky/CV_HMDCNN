import argparse
import copy
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision import datasets
from torchvision.models import resnet18, resnet34

from PIL import ImageOps

try:
    from thop import profile
except ImportError:
    profile = None



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Models
# -----------------------------

class VGG16For32x32(nn.Module):
    """
    VGG16-like network adapted for 32x32 images.

    Original VGG16 is designed for ImageNet 224x224 and has a large classifier.
    For CIFAR/SVHN/EMNIST-style 32x32 images we keep the VGG16 convolutional
    pattern but use a compact classifier after 1x1 feature maps.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, use_bn: bool = True):
        super().__init__()

        cfg = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512, "M",
            512, 512, 512, "M",
        ]

        layers = []
        c_in = in_channels

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(c_in, v, kernel_size=3, padding=1, bias=not use_bn)
                if use_bn:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                c_in = v

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def make_resnet(model_name: str, num_classes: int) -> nn.Module:
    """
    ResNet adapted for 32x32 images:
    - conv1: 3x3, stride 1
    - no initial maxpool
    """

    if model_name == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
    elif model_name == "resnet34":
        model = resnet34(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "vgg16":
        return VGG16For32x32(num_classes=num_classes, in_channels=3, use_bn=True)

    if model_name in {"resnet18", "resnet34"}:
        return make_resnet(model_name, num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")


# -----------------------------
# Datasets
# -----------------------------

DATASET_INFO = {
    "CIFAR10": {
        "num_classes": 10,
        "val_size": 5000,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "SVHN": {
        "num_classes": 10,
        "val_size": 6000,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    "EMNIST": {
        "num_classes": 47,
        "val_size": 10000,
        "mean": (0.1736, 0.1736, 0.1736),
        "std": (0.3317, 0.3317, 0.3317),
    },
}


def emnist_fix_orientation(img):
    """
    Torchvision EMNIST images are commonly displayed rotated/flipped.
    This transform fixes the visual orientation.
    """
    return ImageOps.mirror(img.rotate(-90))


def get_transforms(dataset_name: str, train: bool):
    info = DATASET_INFO[dataset_name]
    mean, std = info["mean"], info["std"]

    if dataset_name == "CIFAR10":
        if train:
            return T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    if dataset_name == "SVHN":
        # В статье для SVHN указывается стандартный вариант без data augmentation.
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    if dataset_name == "EMNIST":
        base = [
            T.Lambda(emnist_fix_orientation),
            T.Resize((32, 32)),
            T.Grayscale(num_output_channels=3),
        ]

        if train:
            return T.Compose(base + [
                T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        return T.Compose(base + [
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_raw_dataset(dataset_name: str, root: str, train: bool, transform):
    root = str(root)

    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )

    if dataset_name == "SVHN":
        return datasets.SVHN(
            root=root,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )

    if dataset_name == "EMNIST":
        return datasets.EMNIST(
            root=root,
            split="balanced",
            train=train,
            download=True,
            transform=transform,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def make_loaders(
    dataset_name: str,
    root: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset_name = dataset_name.upper()
    if dataset_name == "EMNIST-BALANCED":
        dataset_name = "EMNIST"

    info = DATASET_INFO[dataset_name]
    num_classes = info["num_classes"]

    train_transform = get_transforms(dataset_name, train=True)
    eval_transform = get_transforms(dataset_name, train=False)

    full_train_for_train_transform = build_raw_dataset(
        dataset_name, root, train=True, transform=train_transform
    )
    full_train_for_eval_transform = build_raw_dataset(
        dataset_name, root, train=True, transform=eval_transform
    )
    test_dataset = build_raw_dataset(
        dataset_name, root, train=False, transform=eval_transform
    )

    n = len(full_train_for_train_transform)
    val_size = min(info["val_size"], n // 5)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if quick:
        train_indices = train_indices[: min(len(train_indices), 5000)]
        val_indices = val_indices[: min(len(val_indices), 1000)]
        test_indices = np.arange(len(test_dataset))[:1000]
        test_dataset = Subset(test_dataset, test_indices)

    train_dataset = Subset(full_train_for_train_transform, train_indices)
    val_dataset = Subset(full_train_for_eval_transform, val_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, num_classes


# -----------------------------
# Metrics: MMac and model size
# -----------------------------

def model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def compute_mmac(model: nn.Module, device: torch.device) -> float:
    if profile is None:
        return float("nan")

    model.eval()
    dummy = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    return macs / 1e6


# -----------------------------
# Train / Eval
# -----------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp: bool,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    return avg_loss, acc


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device,
    use_amp: bool,
):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    milestones = sorted(set([
        max(1, int(0.50 * epochs)),
        max(1, int(0.75 * epochs)),
    ]))

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1,
    )

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        print(
            f"epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={100 * train_acc:.2f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={100 * val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    results = []

    for dataset_name in args.datasets:
        dataset_name = dataset_name.upper()
        if dataset_name == "EMNIST-BALANCED":
            dataset_name = "EMNIST"

        train_loader, val_loader, test_loader, num_classes = make_loaders(
            dataset_name=dataset_name,
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            quick=args.quick,
        )

        for model_name in args.models:
            model_name = model_name.lower()

            print("\n" + "=" * 80)
            print(f"Dataset: {dataset_name} | Model: {model_name}")
            print("=" * 80)

            model = build_model(model_name, num_classes=num_classes).to(device)

            mmac = compute_mmac(model, device)
            size_mb = model_size_mb(model)

            train_stats = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                use_amp=args.amp and device.type == "cuda",
            )

            test_error = 100.0 * (1.0 - train_stats["test_acc"])

            row = {
                "Dataset": dataset_name,
                "Method": model_name,
                "MMac": round(mmac, 2),
                "Size (MB)": round(size_mb, 2),
                "Test Error (%)": round(test_error, 2),
                "Best Val Acc (%)": round(100.0 * train_stats["best_val_acc"], 2),
                "Test Acc (%)": round(100.0 * train_stats["test_acc"], 2),
            }

            results.append(row)

            df = pd.DataFrame(results)
            print("\nCurrent results:")
            print(df.to_string(index=False))

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

    final_df = pd.DataFrame(results)

    print("\nFinal Table:")
    print(final_df[["Dataset", "Method", "MMac", "Size (MB)", "Test Error (%)"]].to_string(index=False))

    print(f"\nSaved to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CIFAR10", "SVHN", "EMNIST"],
        choices=["CIFAR10", "SVHN", "EMNIST", "EMNIST-BALANCED"],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["vgg16", "resnet18"],
        choices=["vgg16", "resnet18", "resnet34"],
    )

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./baseline_results.csv")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision on CUDA.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small subsets for debugging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)