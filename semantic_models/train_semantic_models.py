from pathlib import Path
import argparse
import random
import time
import csv
import json

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# ============================================================
# PATH 
# ============================================================

PROJECT_ROOT = Path.home() / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"

RESULTS_ROOT = (
    Path.home()
    / "ExoDisc_results"
    / "semantic_benchmark"
)

NUM_CLASSES = 13

CLASSES = {
    0: "background",
    1: "aspirator",
    2: "burr",
    3: "retractor",
    4: "spatula",
    5: "forceps",
    6: "scalpel",
    7: "curettes",
    8: "electrocautery",
    9: "dura",
    10: "ligament",
    11: "herniation",
    12: "disc",
}

FOLDS = {
    "fold1": {"train": ["op2", "op3", "op4"], "val": ["op1"]},
    "fold2": {"train": ["op1", "op3", "op4"], "val": ["op2"]},
    "fold3": {"train": ["op1", "op2", "op4"], "val": ["op3"]},
    "fold4": {"train": ["op1", "op2", "op3"], "val": ["op4"]},
}


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_mask_values(mask_path):
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    values = np.unique(mask)
    bad = values[(values < 0) | (values >= NUM_CLASSES)]
    return values, bad


def save_classes(out_dir):
    with open(out_dir / "classes.txt", "w") as f:
        for idx, name in CLASSES.items():
            f.write(f"{idx}: {name}\n")


# ============================================================
# DATASET
# ============================================================

class ExoDiscSemanticDataset(Dataset):
    def __init__(self, ops, imgsz=512, max_images=None):
        self.items = []
        self.imgsz = imgsz

        for op in ops:
            image_dir = DATASET_ROOT / op / "images"
            mask_dir = DATASET_ROOT / op / "masks_semantic"

            if not image_dir.exists():
                raise FileNotFoundError(f"Missing image dir: {image_dir}")

            if not mask_dir.exists():
                raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

            images = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                images.extend(image_dir.glob(ext))

            images = sorted(images)

            for img_path in images:
                mask_path = mask_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    self.items.append((img_path, mask_path))

        if max_images is not None:
            random.shuffle(self.items)
            self.items = self.items[:max_images]

        if len(self.items) == 0:
            raise RuntimeError(f"Dataset vuoto per ops={ops}")

        print(f"Dataset ops={ops}: {len(self.items)} immagini")

        # controllo veloce prime maschere
        for _, mask_path in self.items[:20]:
            values, bad = check_mask_values(mask_path)
            if len(bad) > 0:
                raise RuntimeError(f"Valori maschera fuori range in {mask_path}: {bad}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Immagine non leggibile: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return image, mask


# ============================================================
# MODELS
# ============================================================

def build_model(model_name):
    if model_name == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )

    if model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )

    if model_name == "segformerb2":
        return smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        )

    raise ValueError(f"Modello non riconosciuto: {model_name}")


# ============================================================
# LOSS / METRICS
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice.mean()


def compute_iou_from_confusion(confusion):
    ious = []

    for cls in range(NUM_CLASSES):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp

        denom = tp + fp + fn

        if denom == 0:
            ious.append(np.nan)
        else:
            ious.append(tp / denom)

    return np.array(ious, dtype=np.float32)


def update_confusion_matrix(confusion, pred, target):
    pred = pred.flatten()
    target = target.flatten()

    valid = (target >= 0) & (target < NUM_CLASSES)
    pred = pred[valid]
    target = target[valid]

    inds = NUM_CLASSES * target + pred
    cm = np.bincount(inds, minlength=NUM_CLASSES ** 2)
    cm = cm.reshape(NUM_CLASSES, NUM_CLASSES)

    confusion += cm


# ============================================================
# TRAIN / VAL
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, ce_loss, dice_loss, device, use_amp):
    model.train()

    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss, device, use_amp):
    model.eval()

    total_loss = 0.0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        for p, t in zip(preds.cpu().numpy(), masks.cpu().numpy()):
            update_confusion_matrix(confusion, p, t)

    class_ious = compute_iou_from_confusion(confusion)

    # mIoU senza background
    miou_no_bg = np.nanmean(class_ious[1:])

    # mIoU con background, se ti serve
    miou_with_bg = np.nanmean(class_ious)

    return total_loss / len(loader), miou_no_bg, miou_with_bg, class_ious


def run_training(
    model_name,
    fold_name,
    epochs,
    imgsz,
    batch,
    lr,
    num_workers,
    max_train,
    max_val,
    seed,
    use_amp,
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 100)
    print(f"MODEL: {model_name} | FOLD: {fold_name}")
    print(f"DEVICE: {device}")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print("=" * 100)

    fold = FOLDS[fold_name]

    train_ds = ExoDiscSemanticDataset(
        ops=fold["train"],
        imgsz=imgsz,
        max_images=max_train,
    )

    val_ds = ExoDiscSemanticDataset(
        ops=fold["val"],
        imgsz=imgsz,
        max_images=max_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    model = build_model(model_name).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(NUM_CLASSES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = RESULTS_ROOT / model_name / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_classes(out_dir)

    config = {
        "model": model_name,
        "fold": fold_name,
        "train_ops": fold["train"],
        "val_ops": fold["val"],
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr": lr,
        "num_workers": num_workers,
        "num_classes": NUM_CLASSES,
        "seed": seed,
        "use_amp": use_amp,
        "dataset_root": str(DATASET_ROOT),
        "results_root": str(RESULTS_ROOT),
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    csv_path = out_dir / "results.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_miou = -1.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "mIoU_no_background",
            "mIoU_with_background",
        ]
        header += [f"IoU_{idx}_{CLASSES[idx]}" for idx in range(NUM_CLASSES)]
        writer.writerow(header)

        for epoch in range(1, epochs + 1):
            start = time.time()

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                ce_loss=ce_loss,
                dice_loss=dice_loss,
                device=device,
                use_amp=use_amp,
            )

            val_loss, miou_no_bg, miou_with_bg, class_ious = validate(
                model=model,
                loader=val_loader,
                ce_loss=ce_loss,
                dice_loss=dice_loss,
                device=device,
                use_amp=use_amp,
            )

            row = [
                epoch,
                train_loss,
                val_loss,
                miou_no_bg,
                miou_with_bg,
            ]
            row += list(class_ious)
            writer.writerow(row)
            f.flush()

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_miou_no_background": best_miou,
                    "classes": CLASSES,
                    "config": config,
                },
                last_path,
            )

            if miou_no_bg > best_miou:
                best_miou = miou_no_bg

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_miou_no_background": best_miou,
                        "classes": CLASSES,
                        "config": config,
                    },
                    best_path,
                )

            elapsed = time.time() - start

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mIoU_no_bg={miou_no_bg:.4f} | "
                f"mIoU_with_bg={miou_with_bg:.4f} | "
                f"time={elapsed:.1f}s"
            )

    print(f"BEST mIoU_no_background: {best_miou:.4f}")
    print(f"SAVED IN: {out_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["unetpp", "deeplabv3plus", "segformerb2"],
        choices=["unetpp", "deeplabv3plus", "segformerb2"],
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        default=["fold1", "fold2", "fold3", "fold4"],
        choices=["fold1", "fold2", "fold3", "fold4"],
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)

    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    use_amp = not args.no_amp

    print("Semantic benchmark training")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"RESULTS_ROOT: {RESULTS_ROOT}")
    print(f"Models: {args.models}")
    print(f"Folds: {args.folds}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"AMP: {use_amp}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        for fold_name in args.folds:
            run_training(
                model_name=model_name,
                fold_name=fold_name,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                lr=args.lr,
                num_workers=args.num_workers,
                max_train=args.max_train,
                max_val=args.max_val,
                seed=args.seed,
                use_amp=use_amp,
            )

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
