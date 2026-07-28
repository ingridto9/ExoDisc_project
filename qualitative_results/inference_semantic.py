#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


# ============================================================
# CONFIGURAZIONE GENERALE
# ============================================================

HOME = Path.home()
PROJECT_ROOT = HOME / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"
RESULTS_ROOT = HOME / "ExoDisc_results" / "semantic_benchmark"
QUALITATIVE_ROOT = HOME / "ExoDisc_results" / "qualitative_inference_comparison"

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

# Stessa palette BGR usata negli altri script.
CLASS_COLORS = {
    "aspirator": (255, 120, 0),
    "burr": (0, 180, 255),
    "retractor": (190, 0, 255),
    "spatula": (255, 0, 120),
    "forceps": (0, 255, 255),
    "scalpel": (80, 255, 80),
    "curettes": (255, 80, 80),
    "electrocautery": (0, 100, 255),
    "dura": (255, 180, 180),
    "ligament": (120, 255, 120),
    "herniation": (180, 180, 255),
    "disc": (120, 120, 255),
}

MODEL_CONFIGS = {
    "unetpp": {
        "folder": "unetpp",
        "display_name": "UNet++",
        "architecture": "unetpp",
        "encoder": "resnet34",
    },
    "deeplabv3plus": {
        "folder": "deeplabv3plus",
        "display_name": "DeepLabV3+",
        "architecture": "deeplabv3plus",
        "encoder": "resnet34",
    },
    "segformerb2": {
        "folder": "segformerb2",
        "display_name": "SegFormer-B2",
        "architecture": "segformer",
        "encoder": "mit_b2",
    },
}

VALID_OPS = ("op1", "op2", "op3", "op4")
VALID_FOLDS = ("fold1", "fold2", "fold3", "fold4")


# ============================================================
# IMAGE SELECTION
# ============================================================

def parse_image_names(values: Sequence[str]) -> List[str]:
    """Accept image names separated by spaces and/or commas."""
    names: List[str] = []

    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                names.append(token)

    if not names:
        raise argparse.ArgumentTypeError(
            "At least one image name must be provided with --images."
        )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(
            "Duplicate image names: " + ", ".join(duplicates)
        )

    return names


def resolve_requested_images(
    image_names: Sequence[str],
    images_dir: Path,
) -> List[dict]:
    """Resolve each requested image by its exact file name."""
    resolved: List[dict] = []
    missing: List[str] = []

    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.is_file():
            missing.append(image_name)
            continue

        resolved.append({
            "file_name": image_name,
            "image_path": image_path,
        })

    if missing:
        raise FileNotFoundError(
            "The following images were not found in "
            f"{images_dir}:\n  - " + "\n  - ".join(missing)
        )

    return resolved


# ============================================================
# UTILS
# ============================================================

def create_contact_sheet(image_paths, output_path, columns=3, thumb_width=640):
    if not image_paths:
        return

    tiles = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb_height = int(image.shape[0] * scale)
        image = cv2.resize(
            image,
            (thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )
        tiles.append(image)

    if not tiles:
        return

    tile_height = max(tile.shape[0] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full(
        (rows * tile_height, columns * thumb_width, 3),
        255,
        dtype=np.uint8,
    )

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        height, width = tile.shape[:2]
        y0 = row * tile_height
        x0 = col * thumb_width
        sheet[y0:y0 + height, x0:x0 + width] = tile

    cv2.imwrite(str(output_path), sheet)


def build_color_mask(semantic_mask):
    color_mask = np.zeros(
        (semantic_mask.shape[0], semantic_mask.shape[1], 3),
        dtype=np.uint8,
    )

    for class_id, class_name in CLASSES.items():
        if class_id == 0:
            continue
        color_mask[semantic_mask == class_id] = CLASS_COLORS[class_name]

    return color_mask


def build_model(model_name):
    cfg = MODEL_CONFIGS[model_name]

    if cfg["architecture"] == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=cfg["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

    if cfg["architecture"] == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=cfg["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

    if cfg["architecture"] == "segformer":
        return smp.Segformer(
            encoder_name=cfg["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

    raise ValueError(f"Unsupported architecture: {cfg['architecture']}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format: {checkpoint_path}"
        )

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]
        cleaned_state_dict[clean_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return checkpoint


def infer_one_image(model, image_bgr, imgsz, device, use_amp):
    original_height, original_width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (imgsz, imgsz),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(device, non_blocking=True)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(tensor)
        else:
            logits = model(tensor)

        prediction = torch.argmax(logits, dim=1)[0]

    semantic_small = prediction.detach().cpu().numpy().astype(np.uint8)
    semantic_mask = cv2.resize(
        semantic_small,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return semantic_mask


# ============================================================
# INFERENZA
# ============================================================

def run_model(
    model_name,
    op_name,
    fold_name,
    image_names,
    imgsz,
    alpha,
    device_name,
    use_amp,
    contact_columns,
):
    cfg = MODEL_CONFIGS[model_name]

    images_dir = DATASET_ROOT / op_name / "images"
    checkpoint_path = RESULTS_ROOT / cfg["folder"] / fold_name / "best.pt"
    output_dir = QUALITATIVE_ROOT / op_name / model_name

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; using CPU.")
        device = torch.device("cpu")
        use_amp = False
    else:
        device = torch.device(device_name)

    masks_dir = output_dir / "semantic_masks"
    color_masks_dir = output_dir / "color_masks"
    overlays_dir = output_dir / "overlays"

    for directory in [masks_dir, color_masks_dir, overlays_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    resolved_images = resolve_requested_images(
        image_names=image_names,
        images_dir=images_dir,
    )

    print("\n" + "=" * 90)
    print(f"Model:    {cfg['display_name']}")
    print(f"Operation:   {op_name}")
    print(f"Fold:       {fold_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images:   {images_dir}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    print("=" * 90)

    print("\nSelected images:")
    for item in resolved_images:
        print(f"  {item['file_name']} -> {item['image_path']}")

    model = build_model(model_name).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    manifest = []
    overlay_paths = []

    for position, image_info in enumerate(resolved_images, start=1):
        image_path = Path(image_info["image_path"])
        file_name = str(image_info["file_name"])
        stem = image_path.stem

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Unable to read: {image_path}")

        semantic_mask = infer_one_image(
            model=model,
            image_bgr=image_bgr,
            imgsz=imgsz,
            device=device,
            use_amp=use_amp,
        )

        # Non esiste una classe lamina nell'output canonico a 13 classi.
        color_mask = build_color_mask(semantic_mask)
        overlay = cv2.addWeighted(image_bgr, 1.0, color_mask, alpha, 0.0)

        label = f"{op_name} - {file_name} - {cfg['display_name']}"
        cv2.rectangle(overlay, (0, 0), (900, 52), (0, 0, 0), thickness=-1)
        cv2.putText(
            overlay,
            label,
            (15, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        semantic_path = masks_dir / f"{stem}_mask.png"
        color_path = color_masks_dir / f"{stem}_color_mask.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"

        if not cv2.imwrite(str(semantic_path), semantic_mask):
            raise RuntimeError(f"Unable to save: {semantic_path}")
        if not cv2.imwrite(str(color_path), color_mask):
            raise RuntimeError(f"Unable to save: {color_path}")
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Unable to save: {overlay_path}")

        present_ids = [
            int(class_id)
            for class_id in np.unique(semantic_mask)
            if int(class_id) != 0
        ]
        present_classes = [CLASSES[class_id] for class_id in present_ids]

        overlay_paths.append(overlay_path)
        manifest.append({
            "position": position,
            "file_name": file_name,
            "image_path": str(image_path),
            "semantic_mask": str(semantic_path),
            "color_mask": str(color_path),
            "overlay": str(overlay_path),
            "present_class_ids": present_ids,
            "present_classes": present_classes,
        })

        print(
            f"[{position:02d}/{len(resolved_images)}] "
            f"{file_name}: "
            f"{', '.join(present_classes) if present_classes else 'background only'}"
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": model_name,
                "display_name": cfg["display_name"],
                "patient": op_name,
                "fold": fold_name,
                "checkpoint": str(checkpoint_path),
                "images_dir": str(images_dir),
                "requested_images": list(image_names),
                "imgsz": imgsz,
                "alpha": alpha,
                "classes": CLASSES,
                "checkpoint_epoch": (
                    checkpoint.get("epoch")
                    if isinstance(checkpoint, dict)
                    else None
                ),
                "images": manifest,
            },
            file,
            indent=2,
        )

    contact_sheet_path = (
        output_dir
        / f"{op_name}_{model_name}_{len(resolved_images)}_images.jpg"
    )
    create_contact_sheet(
        overlay_paths,
        contact_sheet_path,
        columns=contact_columns,
    )

    print("\nInference completed.")
    print(f"Manifest:      {manifest_path}")
    print(f"Contact sheet: {contact_sheet_path}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Qualitative inference for the semantic models on selected images "
            "from op1-op4, using exact image file names."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["unetpp", "deeplabv3plus", "segformerb2", "all"],
        help="Modello da eseguire oppure 'all' per eseguirli tutti.",
    )
    parser.add_argument(
        "--op",
        required=True,
        choices=VALID_OPS,
        help="Paziente/operazione da elaborare.",
    )
    parser.add_argument(
        "--fold",
        required=True,
        choices=VALID_FOLDS,
        help="Fold contenente il checkpoint da usare.",
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        help=(
            "Exact image file names, separated by spaces or commas. "
            "Example: --images op1_0123.png op1_0456.png"
        ),
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--contact-columns", type=int, default=3)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than zero.")
    if args.contact_columns <= 0:
        parser.error("--contact-columns must be greater than zero.")

    try:
        image_names = parse_image_names(args.images)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    model_names = (
        list(MODEL_CONFIGS.keys())
        if args.model == "all"
        else [args.model]
    )

    for model_name in model_names:
        run_model(
            model_name=model_name,
            op_name=args.op,
            fold_name=args.fold,
            image_names=image_names,
            imgsz=args.imgsz,
            alpha=args.alpha,
            device_name=args.device,
            use_amp=not args.no_amp,
            contact_columns=args.contact_columns,
        )


if __name__ == "__main__":
    main()