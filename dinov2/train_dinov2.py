#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel

PROJECT_ROOT = Path.home() / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"

RESULTS_ROOT = (
    Path.home()
    / "ExoDisc_results"
    / "semantic_benchmark"
    / "dinov2"
)


FOLDS = {
    "fold1": {"train": ["op2", "op3", "op4"], "val": ["op1"]},
    "fold2": {"train": ["op1", "op3", "op4"], "val": ["op2"]},
    "fold3": {"train": ["op1", "op2", "op4"], "val": ["op3"]},
    "fold4": {"train": ["op1", "op2", "op3"], "val": ["op4"]},
}

CLASS_NAMES = [
    "background",
    "aspirator",
    "burr",
    "retractor",
    "spatula",
    "forceps",
    "scalpel",
    "curettes",
    "electrocautery",
    "dura",
    "ligament",
    "herniation",
    "disc",
]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).float()


def color_jitter(img):
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    return img


class ExoDiscSemanticDataset(Dataset):
    def __init__(self, dataset_root, ops, image_size=518, augment=False):
        self.dataset_root = Path(dataset_root)
        self.ops = ops
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        for op in ops:
            images_dir = self.dataset_root / op / "images"
            masks_dir = self.dataset_root / op / "masks_semantic"

            if not images_dir.exists():
                raise FileNotFoundError(f"Cartella immagini non trovata: {images_dir}")
            if not masks_dir.exists():
                raise FileNotFoundError(f"Cartella maschere non trovata: {masks_dir}")

            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in image_exts:
                    continue

                mask_path = masks_dir / f"{img_path.stem}.png"

                if mask_path.exists():
                    self.samples.append((img_path, mask_path, op))

        if len(self.samples) == 0:
            raise RuntimeError(f"Nessuna coppia immagine/maschera trovata per ops={ops}")

        print(f"Loaded {len(self.samples)} samples from {ops}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, op = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        arr = arr.astype(np.int64)
        arr[(arr < 0) | (arr >= len(CLASS_NAMES))] = 255

        mask = Image.fromarray(arr.astype(np.uint8))

        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            img = color_jitter(img)

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        return {
            "image": normalize_image(img),
            "mask": torch.from_numpy(np.asarray(mask).astype(np.int64)),
        }


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state[:, 1:, :]

        b, n, c = tokens.shape
        h = w = int(n ** 0.5)

        if h * w != n:
            raise RuntimeError(f"Numero patch non quadrato: n={n}")

        feat = tokens.transpose(1, 2).reshape(b, c, h, w)

        logits = self.decoder(feat)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        valid = target != self.ignore_index
        target_clean = target.clone()
        target_clean[~valid] = 0

        target_1h = F.one_hot(target_clean, self.num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()

        probs = probs * valid
        target_1h = target_1h * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_1h, dims)
        cardinality = torch.sum(probs + target_1h, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice.mean()


@torch.no_grad()
def update_confmat(confmat, preds, target, num_classes, ignore_index=255):
    preds = preds.detach().cpu().numpy().astype(np.int64)
    target = target.detach().cpu().numpy().astype(np.int64)

    valid = target != ignore_index
    preds = preds[valid]
    target = target[valid]

    valid2 = (target >= 0) & (target < num_classes)
    preds = preds[valid2]
    target = target[valid2]

    inds = num_classes * target + preds
    cm = np.bincount(
        inds,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    confmat += cm


def compute_metrics(confmat):
    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn

    iou = np.divide(
        tp,
        denom_iou,
        out=np.full_like(tp, np.nan),
        where=denom_iou > 0,
    )

    dice = np.divide(
        2 * tp,
        denom_dice,
        out=np.full_like(tp, np.nan),
        where=denom_dice > 0,
    )

    miou = np.nanmean(iou)
    mdice = np.nanmean(dice)

    return miou, mdice, iou, dice


def run_epoch(
    model,
    loader,
    optimizer,
    ce_loss,
    dice_loss,
    device,
    num_classes,
    train=True,
    amp=True,
):
    if train:
        model.train()
    else:
        model.eval()

    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_batches = 0

    scaler = torch.cuda.amp.GradScaler(
        enabled=(train and amp and device.type == "cuda")
    )

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = model(images)
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = torch.argmax(logits, dim=1)
        update_confmat(confmat, preds, masks, num_classes)

        total_loss += loss.item()
        total_batches += 1

        if step == 1 or step % 50 == 0:
            phase = "train" if train else "val"
            print(
                f"  {phase} step {step}/{len(loader)} | loss={loss.item():.4f}",
                flush=True,
            )

    miou, mdice, iou, dice = compute_metrics(confmat)

    return total_loss / max(total_batches, 1), miou, mdice, iou, dice


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", required=True, choices=["fold1", "fold2", "fold3", "fold4"])
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root containing op1, op2, op3 and op4.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory in which DINOv2 results are saved.",
    )

    parser.add_argument("--model_name", default="facebook/dinov2-base")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    num_classes = len(CLASS_NAMES)

    fold_cfg = FOLDS[args.fold]
    train_ops = fold_cfg["train"]
    val_ops = fold_cfg["val"]

    out_dir = args.results_root / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump(config, f, indent=2)

    with open(out_dir / "classes.txt", "w") as f:
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i},{name}\n")

    print("=" * 70)
    print("DINOv2 semantic segmentation benchmark")
    print("Fold:", args.fold)
    print("Train ops:", train_ops)
    print("Val ops:", val_ops)
    print("Raw root:", args.dataset_root)
    print("Images path pattern: raw/opX/images")
    print("Masks path pattern: raw/opX/masks_semantic")
    print("Classes:", CLASS_NAMES)
    print("Output:", out_dir)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train_ds = ExoDiscSemanticDataset(
        dataset_root=args.dataset_root,
        ops=train_ops,
        image_size=args.image_size,
        augment=True,
    )

    val_ds = ExoDiscSemanticDataset(
        dataset_root=args.dataset_root,
        ops=val_ops,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = DINOv2SegmentationModel(
        model_name=args.model_name,
        num_classes=num_classes,
    ).to(device)

    backbone_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": decoder_params, "lr": args.lr_decoder},
        ],
        weight_decay=args.weight_decay,
    )

    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(num_classes=num_classes, ignore_index=255)

    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        print(f"\nEpoch {epoch:03d}/{args.epochs} - training", flush=True)

        train_loss, train_miou, train_dice, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ce_loss=ce_loss,
            dice_loss=dice_loss,
            device=device,
            num_classes=num_classes,
            train=True,
            amp=not args.no_amp,
        )

        print(f"\nEpoch {epoch:03d}/{args.epochs} - validation", flush=True)

        val_loss, val_miou, val_dice, val_iou_pc, val_dice_pc = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            ce_loss=ce_loss,
            dice_loss=dice_loss,
            device=device,
            num_classes=num_classes,
            train=False,
            amp=not args.no_amp,
        )

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mIoU": train_miou,
            "train_Dice": train_dice,
            "val_loss": val_loss,
            "val_mIoU": val_miou,
            "val_Dice": val_dice,
            "time_sec": elapsed,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "results.csv", index=False)

        print(
            f"Epoch {epoch:03d} summary | "
            f"train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} train_Dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} val_Dice={val_dice:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if val_miou > best_miou:
            best_miou = val_miou

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou,
                    "class_names": CLASS_NAMES,
                    "args": config,
                },
                out_dir / "best_model.pt",
            )

            per_class = []
            for i, name in enumerate(CLASS_NAMES):
                per_class.append(
                    {
                        "class_id": i,
                        "class_name": name,
                        "IoU": val_iou_pc[i],
                        "Dice": val_dice_pc[i],
                    }
                )

            pd.DataFrame(per_class).to_csv(out_dir / "per_class_best.csv", index=False)

            print(f"  New best model saved. best_mIoU={best_miou:.4f}", flush=True)

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "class_names": CLASS_NAMES,
            "args": config,
        },
        out_dir / "last_model.pt",
    )

    print("\nTraining finished.")
    print(f"Best val mIoU: {best_miou:.4f}")
    print(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()