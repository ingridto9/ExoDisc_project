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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


PROJECT_ROOT = Path.home() / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"
RESULTS_ROOT = Path.home() / "ExoDisc_results" / "semantic_benchmark"

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_classes(out_dir):
    with open(out_dir / "classes.txt", "w") as f:
        for idx, name in CLASSES.items():
            f.write(f"{idx}: {name}\n")


def check_mask_values(mask_path):
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    values = np.unique(mask)
    bad = values[(values >= NUM_CLASSES) & (values != 255)]
    return values, bad


class ExoDiscMask2FormerDataset(Dataset):
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

            for img_path in sorted(images):
                mask_path = mask_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    self.items.append((img_path, mask_path))

        if max_images is not None:
            random.shuffle(self.items)
            self.items = self.items[:max_images]

        if len(self.items) == 0:
            raise RuntimeError(f"Dataset vuoto per ops={ops}")

        print(f"Dataset ops={ops}: {len(self.items)} immagini", flush=True)

        for _, mask_path in self.items[:20]:
            _, bad = check_mask_values(mask_path)
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

        return image, mask


def make_mask2former_targets(masks):
    mask_labels = []
    class_labels = []

    for mask in masks:
        labels = np.unique(mask)
        labels = labels[(labels >= 0) & (labels < NUM_CLASSES)]

        binary_masks = []
        class_ids = []

        for label in labels:
            binary = (mask == label).astype(np.float32)
            if binary.sum() == 0:
                continue
            binary_masks.append(torch.from_numpy(binary))
            class_ids.append(int(label))

        if len(binary_masks) == 0:
            binary_masks = [torch.ones_like(torch.from_numpy(mask).float())]
            class_ids = [0]

        mask_labels.append(torch.stack(binary_masks, dim=0))
        class_labels.append(torch.tensor(class_ids, dtype=torch.long))

    return mask_labels, class_labels


def build_collate_fn(processor):
    def collate_fn(batch):
        images, masks = zip(*batch)

        encoded = processor(
            images=list(images),
            return_tensors="pt",
            do_resize=False,
        )

        mask_labels, class_labels = make_mask2former_targets(masks)

        encoded["mask_labels"] = mask_labels
        encoded["class_labels"] = class_labels
        encoded["semantic_masks"] = torch.from_numpy(np.stack(masks, axis=0)).long()

        return encoded

    return collate_fn


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


def outputs_to_semantic_preds(outputs, target_size):
    class_logits = outputs.class_queries_logits
    mask_logits = outputs.masks_queries_logits

    mask_logits = F.interpolate(
        mask_logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    class_probs = torch.softmax(class_logits, dim=-1)[..., :NUM_CLASSES]
    mask_probs = torch.sigmoid(mask_logits)

    semantic_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    preds = torch.argmax(semantic_scores, dim=1)

    return preds


def move_batch_to_device(batch, device):
    out = {}

    for k, v in batch.items():
        if k in ["mask_labels", "class_labels"]:
            out[k] = [x.to(device) for x in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, log_interval):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                pixel_values=batch["pixel_values"],
                pixel_mask=batch.get("pixel_mask", None),
                mask_labels=batch["mask_labels"],
                class_labels=batch["class_labels"],
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % log_interval == 0 or step == 1 or step == len(loader):
            print(
                f"  train step {step}/{len(loader)} | loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, use_amp, log_interval):
    model.eval()

    total_loss = 0.0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        masks = batch["semantic_masks"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                pixel_values=batch["pixel_values"],
                pixel_mask=batch.get("pixel_mask", None),
                mask_labels=batch["mask_labels"],
                class_labels=batch["class_labels"],
            )
            loss = outputs.loss

        total_loss += loss.item()

        preds = outputs_to_semantic_preds(
            outputs=outputs,
            target_size=masks.shape[-2:],
        )

        for p, t in zip(preds.cpu().numpy(), masks.cpu().numpy()):
            update_confusion_matrix(confusion, p, t)

        if step % log_interval == 0 or step == 1 or step == len(loader):
            print(
                f"  val step {step}/{len(loader)} | loss={loss.item():.4f}",
                flush=True,
            )

    class_ious = compute_iou_from_confusion(confusion)
    miou_no_bg = np.nanmean(class_ious[1:])
    miou_with_bg = np.nanmean(class_ious)

    return total_loss / len(loader), miou_no_bg, miou_with_bg, class_ious


def build_model(checkpoint):
    id2label = {i: CLASSES[i] for i in range(NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return model


def run_training(
    fold_name,
    checkpoint,
    epochs,
    imgsz,
    batch,
    lr,
    num_workers,
    max_train,
    max_val,
    seed,
    use_amp,
    log_interval,
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 100, flush=True)
    print(f"MODEL: hf_mask2former | FOLD: {fold_name}", flush=True)
    print(f"CHECKPOINT: {checkpoint}", flush=True)
    print(f"DEVICE: {device}", flush=True)
    print(f"DATASET_ROOT: {DATASET_ROOT}", flush=True)
    print("=" * 100, flush=True)

    fold = FOLDS[fold_name]

    processor = AutoImageProcessor.from_pretrained(checkpoint)

    train_ds = ExoDiscMask2FormerDataset(
        ops=fold["train"],
        imgsz=imgsz,
        max_images=max_train,
    )

    val_ds = ExoDiscMask2FormerDataset(
        ops=fold["val"],
        imgsz=imgsz,
        max_images=max_val,
    )

    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = build_model(checkpoint).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = RESULTS_ROOT / "hf_mask2former" / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_classes(out_dir)

    config = {
        "model": "hf_mask2former",
        "checkpoint": checkpoint,
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

            print(f"\nEpoch {epoch:03d}/{epochs} - training", flush=True)

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                log_interval=log_interval,
            )

            print(f"Epoch {epoch:03d}/{epochs} - validation", flush=True)

            val_loss, miou_no_bg, miou_with_bg, class_ious = validate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                log_interval=max(50, log_interval),
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
                f"time={elapsed:.1f}s",
                flush=True,
            )

    print(f"BEST mIoU_no_background: {best_miou:.4f}", flush=True)
    print(f"SAVED IN: {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        nargs="+",
        default=["fold1", "fold2", "fold3", "fold4"],
        choices=["fold1", "fold2", "fold3", "fold4"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/mask2former-swin-tiny-ade-semantic",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)

    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    use_amp = not args.no_amp

    print("HF Mask2Former training", flush=True)
    print(f"DATASET_ROOT: {DATASET_ROOT}", flush=True)
    print(f"RESULTS_ROOT: {RESULTS_ROOT}", flush=True)
    print(f"Folds: {args.folds}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Image size: {args.imgsz}", flush=True)
    print(f"Batch: {args.batch}", flush=True)
    print(f"LR: {args.lr}", flush=True)
    print(f"AMP: {use_amp}", flush=True)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for fold_name in args.folds:
        run_training(
            fold_name=fold_name,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr=args.lr,
            num_workers=args.num_workers,
            max_train=args.max_train,
            max_val=args.max_val,
            seed=args.seed,
            use_amp=use_amp,
            log_interval=args.log_interval,
        )

    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    main()
