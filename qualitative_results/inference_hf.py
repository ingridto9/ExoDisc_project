#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, Mask2FormerForUniversalSegmentation


# ============================================================
# CONFIGURAZIONE GENERALE
# ============================================================

HOME = Path.home()
PROJECT_ROOT = HOME / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"
RESULTS_ROOT = HOME / "ExoDisc_results" / "semantic_benchmark"
QUALITATIVE_ROOT = HOME / "ExoDisc_results" / "qualitative_inference_comparison"

VALID_OPS = ("op1", "op2", "op3", "op4")
VALID_FOLDS = ("fold1", "fold2", "fold3", "fold4")

CANONICAL_CLASSES: Dict[int, str] = {
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

NAME_TO_CANONICAL_ID = {
    class_name: class_id for class_id, class_name in CANONICAL_CLASSES.items()
}
NUM_CLASSES = len(CANONICAL_CLASSES)

# Palette BGR identica a YOLO11 e agli altri modelli semantici.
STANDARD_ID_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 120, 0),       # aspirator
    2: (0, 180, 255),       # burr
    3: (190, 0, 255),       # retractor
    4: (255, 0, 120),       # spatula
    5: (0, 255, 255),       # forceps
    6: (80, 255, 80),       # scalpel
    7: (255, 80, 80),       # curettes
    8: (0, 100, 255),       # electrocautery
    9: (255, 180, 180),     # dura
    10: (120, 255, 120),    # ligament
    11: (180, 180, 255),    # herniation
    12: (120, 120, 255),    # disc
}

# DINOv2 e' stato addestrato con l'ordine:
# 9=disc, 10=dura, 11=ligament, 12=herniation.
# La maschera numerica DINO viene lasciata invariata; cambiamo solo i colori.
DINO_ID_COLORS = STANDARD_ID_COLORS.copy()
# DINO_ID_COLORS[9] = STANDARD_ID_COLORS[12]   # ID DINO 9 = disc
# DINO_ID_COLORS[10] = STANDARD_ID_COLORS[9]  # ID DINO 10 = dura
# DINO_ID_COLORS[11] = STANDARD_ID_COLORS[10] # ID DINO 11 = ligament
# DINO_ID_COLORS[12] = STANDARD_ID_COLORS[11] # ID DINO 12 = herniation

MODEL_CONFIGS = {
    "dinov2": {
        "display_name": "DINOv2",
        "result_folders": ["dinov2"],
        "checkpoint_names": ["best_model.pt", "best.pt"],
        "hf_model": "facebook/dinov2-base",
        "image_size": 518,
    },
    "mask2former": {
        "display_name": "Mask2Former",
        "result_folders": ["hf_mask2former", "mask2former"],
        "checkpoint_names": ["best.pt", "best_model.pt"],
        "hf_model": "facebook/mask2former-swin-tiny-ade-semantic",
        "image_size": 512,
    },
}


# ============================================================
# MODELLO DINOV2
# ============================================================

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state[:, 1:, :]

        batch, num_tokens, channels = tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError(f"Non-square number of DINOv2 patches: {num_tokens}")

        features = tokens.transpose(1, 2).reshape(batch, channels, side, side)
        logits = self.decoder(features)
        return F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


# ============================================================
# IMAGE SELECTION BY EXACT FILE NAME
# ============================================================

def parse_image_arguments(values: Sequence[str]) -> List[str]:
    """Accept image names separated by spaces and/or commas."""
    parsed: List[str] = []

    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(token)

    if not parsed:
        raise ValueError("At least one image name must be provided with --images.")

    duplicates = sorted({name for name in parsed if parsed.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Duplicate image names were provided: " + ", ".join(duplicates)
        )

    return parsed


def resolve_selected_images(
    requested_names: Sequence[str],
    images_dir: Path,
) -> List[dict]:
    """Resolve images by their exact file name inside the operation folder."""
    selected: List[dict] = []
    missing: List[str] = []

    for file_name in requested_names:
        # Path(...).name prevents a user-supplied directory from escaping images_dir.
        clean_name = Path(file_name).name
        image_path = images_dir / clean_name

        if not image_path.is_file():
            missing.append(clean_name)
            continue

        selected.append({
            "file_name": clean_name,
            "image_path": image_path,
        })

    if missing:
        raise FileNotFoundError(
            "The following images were not found in "
            f"{images_dir}: " + ", ".join(missing)
        )

    print("\nSelected images:", flush=True)
    for item in selected:
        print(f"  {item['file_name']} -> {item['image_path']}", flush=True)

    return selected


# ============================================================
# UTILS MODELLI E OUTPUT
# ============================================================

def build_checkpoint_candidates(model_name: str, fold: str) -> List[Path]:
    config = MODEL_CONFIGS[model_name]
    candidates: List[Path] = []
    for folder in config["result_folders"]:
        for checkpoint_name in config["checkpoint_names"]:
            candidates.append(RESULTS_ROOT / folder / fold / checkpoint_name)
    return candidates


def resolve_checkpoint(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Attempted paths:\n{attempted}")


def load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint format: {path}")
    return checkpoint


def normalize_class_names(raw_names) -> Optional[List[str]]:
    if isinstance(raw_names, dict):
        pairs = sorted(
            ((int(key), str(value)) for key, value in raw_names.items()),
            key=lambda pair: pair[0],
        )
        return [name for _, name in pairs]
    if isinstance(raw_names, (list, tuple)):
        return [str(name) for name in raw_names]
    return None


def remap_to_canonical(raw_mask: np.ndarray, source_class_names: Sequence[str]) -> np.ndarray:
    canonical = np.zeros(raw_mask.shape, dtype=np.uint8)
    for source_id, class_name in enumerate(source_class_names):
        normalized_name = class_name.strip().lower()
        if normalized_name == "lamina":
            continue
        if normalized_name not in NAME_TO_CANONICAL_ID:
            raise RuntimeError(f"Unknown class in checkpoint: {normalized_name}")
        canonical[raw_mask == source_id] = NAME_TO_CANONICAL_ID[normalized_name]
    return canonical


def build_color_mask(semantic_mask: np.ndarray, model_name: str) -> np.ndarray:
    color_mask = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    palette = DINO_ID_COLORS if model_name == "dinov2" else STANDARD_ID_COLORS

    for class_id, color in palette.items():
        if class_id == 0:
            continue
        color_mask[semantic_mask == class_id] = color

    return color_mask


def make_overlay(
    image_bgr: np.ndarray,
    semantic_mask: np.ndarray,
    alpha: float,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    color_mask = build_color_mask(semantic_mask, model_name)
    foreground = semantic_mask != 0
    overlay = image_bgr.copy()
    if np.any(foreground):
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0.0)
        overlay[foreground] = blended[foreground]
    return overlay, color_mask


def add_image_label(
    image: np.ndarray,
    image_name: str,
    display_name: str,
) -> np.ndarray:
    output = image.copy()
    text = f"{image_name} - {display_name}"
    cv2.rectangle(output, (0, 0), (850, 52), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        text,
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def create_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    columns: int = 3,
    thumb_width: int = 640,
) -> None:
    if not image_paths:
        return

    tiles = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb_height = int(round(image.shape[0] * scale))
        tiles.append(
            cv2.resize(
                image,
                (thumb_width, thumb_height),
                interpolation=cv2.INTER_AREA,
            )
        )

    if not tiles:
        return

    tile_height = max(tile.shape[0] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full(
        (rows * tile_height, columns * thumb_width, 3),
        255,
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y0 = row * tile_height
        x0 = column * thumb_width
        height, width = tile.shape[:2]
        sheet[y0:y0 + height, x0:x0 + width] = tile

    cv2.imwrite(str(output_path), sheet)


# ============================================================
# DINOV2
# ============================================================

def load_dinov2(config: dict, fold: str, device: torch.device):
    checkpoint_path = resolve_checkpoint(build_checkpoint_candidates("dinov2", fold))
    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args) if hasattr(saved_args, "__dict__") else {}

    hf_model = saved_args.get("model_name", config["hf_model"])
    image_size = int(saved_args.get("image_size", config["image_size"]))

    source_names = normalize_class_names(checkpoint.get("class_names"))
    if source_names is None:
        source_names = [
            "background", "aspirator", "burr", "retractor", "spatula",
            "forceps", "scalpel", "curettes", "electrocautery",
            "disc", "dura", "ligament", "herniation",
        ]

    model = DINOv2SegmentationModel(hf_model, len(source_names)).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("DINOv2 checkpoint is missing 'model_state_dict'")

    cleaned_state_dict = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    print(f"DINOv2 checkpoint: {checkpoint_path}", flush=True)
    print(f"DINOv2 backbone:   {hf_model}", flush=True)
    print(f"DINOv2 class order: {source_names}", flush=True)

    return model, image_size, source_names, checkpoint_path


def preprocess_dinov2(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(
        image_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    array = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)


@torch.inference_mode()
def predict_dinov2(
    model: nn.Module,
    image_bgr: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    original_height, original_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess_dinov2(image_rgb, image_size).to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(tensor)

    raw_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(
        raw_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )


# ============================================================
# MASK2FORMER
# ============================================================

def build_mask2former_model(hf_model: str, source_names: Sequence[str]):
    id2label = {index: name for index, name in enumerate(source_names)}
    label2id = {name: index for index, name in id2label.items()}
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        hf_model,
        num_labels=len(source_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def load_mask2former(config: dict, fold: str, device: torch.device):
    checkpoint_path = resolve_checkpoint(
        build_checkpoint_candidates("mask2former", fold)
    )
    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    saved_config = checkpoint.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = vars(saved_config) if hasattr(saved_config, "__dict__") else {}

    hf_model = saved_config.get("checkpoint", config["hf_model"])
    image_size = int(saved_config.get("imgsz", config["image_size"]))

    source_names = normalize_class_names(checkpoint.get("classes"))
    if source_names is None:
        source_names = [CANONICAL_CLASSES[index] for index in range(NUM_CLASSES)]

    processor = AutoImageProcessor.from_pretrained(hf_model)
    model = build_mask2former_model(hf_model, source_names).to(device)

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Mask2Former checkpoint is missing 'model_state_dict'")

    cleaned_state_dict = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    print(f"Mask2Former checkpoint: {checkpoint_path}", flush=True)
    print(f"Mask2Former base model:  {hf_model}", flush=True)
    print(f"Mask2Former class order: {source_names}", flush=True)

    return model, processor, image_size, source_names, checkpoint_path


@torch.inference_mode()
def predict_mask2former(
    model,
    processor,
    image_bgr: np.ndarray,
    image_size: int,
    source_names: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    original_height, original_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    encoded = processor(images=[resized], return_tensors="pt", do_resize=False)
    pixel_values = encoded["pixel_values"].to(device)
    pixel_mask = encoded.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    class_logits = outputs.class_queries_logits
    mask_logits = F.interpolate(
        outputs.masks_queries_logits,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    class_probs = torch.softmax(class_logits, dim=-1)[..., :len(source_names)]
    mask_probs = torch.sigmoid(mask_logits)
    semantic_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    raw_mask = torch.argmax(semantic_scores, dim=1)[0].cpu().numpy().astype(np.uint8)
    raw_mask = cv2.resize(
        raw_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return remap_to_canonical(raw_mask, source_names)


# ============================================================
# ESECUZIONE
# ============================================================

def run_model(
    model_name: str,
    op: str,
    fold: str,
    selected_images: Sequence[dict],
    images_dir: Path,
    output_root: Path,
    alpha: float,
    device: torch.device,
) -> None:
    config = MODEL_CONFIGS[model_name]
    display_name = config["display_name"]
    output_dir = output_root / model_name

    masks_dir = output_dir / "semantic_masks"
    colors_dir = output_dir / "color_masks"
    overlays_dir = output_dir / "overlays"
    for folder in (masks_dir, colors_dir, overlays_dir):
        folder.mkdir(parents=True, exist_ok=True)

    if model_name == "dinov2":
        model, image_size, source_names, checkpoint_path = load_dinov2(
            config, fold, device
        )

        def predictor(image):
            return predict_dinov2(model, image, image_size, device)

    elif model_name == "mask2former":
        model, processor, image_size, source_names, checkpoint_path = load_mask2former(
            config, fold, device
        )

        def predictor(image):
            return predict_mask2former(
                model,
                processor,
                image,
                image_size,
                source_names,
                device,
            )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print("\n" + "=" * 90, flush=True)
    print(f"Modello:    {display_name}", flush=True)
    print(f"Paziente:   {op}", flush=True)
    print(f"Fold:       {fold}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Immagini:   {images_dir}", flush=True)
    print(f"Output:     {output_dir}", flush=True)
    print(f"Device:     {device}", flush=True)
    print("=" * 90, flush=True)

    manifest = {
        "model": model_name,
        "display_name": display_name,
        "patient": op,
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "images_dir": str(images_dir),
        "alpha": alpha,
        "canonical_classes": CANONICAL_CLASSES,
        "source_class_order": source_names,
        "visualization_palette": (
            "dinov2_source_id_palette"
            if model_name == "dinov2"
            else "canonical_id_palette"
        ),
        "images": [],
    }
    overlay_paths: List[Path] = []

    for position, item in enumerate(selected_images, start=1):
        file_name = str(item["file_name"])
        image_path = Path(item["image_path"])
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Unreadable image: {image_path}")

        semantic_mask = predictor(image_bgr)
        overlay, color_mask = make_overlay(
            image_bgr,
            semantic_mask,
            alpha,
            model_name,
        )
        overlay = add_image_label(overlay, file_name, display_name)

        stem = image_path.stem
        mask_path = masks_dir / f"{stem}_mask.png"
        color_path = colors_dir / f"{stem}_color_mask.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"

        cv2.imwrite(str(mask_path), semantic_mask)
        cv2.imwrite(str(color_path), color_mask)
        cv2.imwrite(str(overlay_path), overlay)
        overlay_paths.append(overlay_path)

        predicted_ids = sorted(
            int(value) for value in np.unique(semantic_mask) if int(value) != 0
        )
        if model_name == "dinov2":
            predicted_classes = [
                source_names[class_id]
                for class_id in predicted_ids
                if class_id < len(source_names)
            ]
        else:
            predicted_classes = [
                CANONICAL_CLASSES[class_id]
                for class_id in predicted_ids
                if class_id in CANONICAL_CLASSES
            ]

        manifest["images"].append({
            "position": position,
            "file_name": file_name,
            "image": str(image_path),
            "semantic_mask": str(mask_path),
            "color_mask": str(color_path),
            "overlay": str(overlay_path),
            "predicted_class_ids": predicted_ids,
            "predicted_classes": predicted_classes,
        })

        print(
            f"[{display_name}] {position:02d}/{len(selected_images)} "
            f"{file_name}: {predicted_classes}",
            flush=True,
        )

    contact_sheet_path = output_dir / (
        f"{op}_{model_name}_{len(selected_images)}_images.jpg"
    )
    create_contact_sheet(overlay_paths, contact_sheet_path, columns=3)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"\n{display_name} completed.", flush=True)
    print(f"Manifest:      {manifest_path}", flush=True)
    print(f"Contact sheet: {contact_sheet_path}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qualitative inference for DINOv2 and Mask2Former on selected "
            "ExoDisc images resolved by exact file name."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["dinov2", "mask2former", "all"],
        help="Model to run, or 'all' to run both models.",
    )
    parser.add_argument(
        "--op",
        required=True,
        choices=VALID_OPS,
        help="Operation to process.",
    )
    parser.add_argument(
        "--fold",
        required=True,
        choices=VALID_FOLDS,
        help="Fold from which the trained checkpoint is loaded.",
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        help=(
            "Exact image file names, separated by spaces or commas. "
            "Example: --images op1_0012.png op1_0345.png"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Examples: cuda:0, cuda:1, cpu.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    requested_names = parse_image_arguments(args.images)

    images_dir = DATASET_ROOT / args.op / "images"
    output_root = QUALITATIVE_ROOT / args.op

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    selected_images = resolve_selected_images(
        requested_names=requested_names,
        images_dir=images_dir,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    models = ["dinov2", "mask2former"] if args.model == "all" else [args.model]
    for model_name in models:
        run_model(
            model_name=model_name,
            op=args.op,
            fold=args.fold,
            selected_images=selected_images,
            images_dir=images_dir,
            output_root=output_root,
            alpha=args.alpha,
            device=device,
        )

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()