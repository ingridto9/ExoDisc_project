#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry


# ============================================================
# CONFIGURAZIONE FISSA
# ============================================================

HOME = Path.home()
DATASET_ROOT = HOME / "ExoDisc_project" / "ExoDisc"
CHECKPOINT = (
    HOME
    / "ExoDisc_project"
    / "checkpoints"
    / "medsam"
    / "medsam_vit_b.pth"
)
QUALITATIVE_ROOT = (
    HOME
    / "ExoDisc_results"
    / "qualitative_inference_comparison"
)

VALID_OPS = ("op1", "op2", "op3", "op4")

# COCO IDs originali del dataset.
COCO_CLASS_NAMES = {
    1: "disc",
    2: "dura",
    3: "ligament",
    4: "lamina",
    5: "aspirator",
    6: "burr",
    7: "retractor",
    8: "spatula",
    9: "forceps",
    10: "scalpel",
    11: "curettes",
    12: "electrocautery",
    13: "herniation",
}

# Ordine canonico usato da tutti gli altri modelli semantici.
CANONICAL_CLASS_NAMES = {
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
    class_name: class_id
    for class_id, class_name in CANONICAL_CLASS_NAMES.items()
}

# Palette BGR identica a YOLO11 e agli altri modelli semantici.
CLASS_COLORS_BY_NAME = {
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

CLASS_COLORS = {
    class_id: CLASS_COLORS_BY_NAME.get(class_name, (0, 0, 0))
    for class_id, class_name in CANONICAL_CLASS_NAMES.items()
}

VALID_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ============================================================
# IMAGE SELECTION BY EXACT FILE NAME
# ============================================================


def parse_image_names(raw_values: List[str]) -> List[str]:
    """Accept image names separated by spaces, commas, or both."""
    names: List[str] = []

    for raw_value in raw_values:
        for token in raw_value.split(","):
            name = token.strip()
            if not name:
                continue
            if Path(name).name != name:
                raise argparse.ArgumentTypeError(
                    f"Invalid image name {name!r}. Provide only the file name, "
                    "not a directory path."
                )
            names.append(name)

    if not names:
        raise argparse.ArgumentTypeError(
            "At least one image name must be provided with --images."
        )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(
            "Duplicate image names in --images: " + ", ".join(duplicates)
        )

    return names


def build_coco_indices(coco: dict):
    """Build exact file-name and annotation indices from a COCO dictionary."""
    images_by_file_name: Dict[str, dict] = {}
    annotations_by_image: Dict[int, List[dict]] = {}

    for image_item in coco.get("images", []):
        coco_id = int(image_item["id"])
        file_name = Path(str(image_item.get("file_name", ""))).name
        if not file_name:
            continue
        if file_name in images_by_file_name:
            raise RuntimeError(
                "The COCO file contains duplicate image file names: "
                f"{file_name}"
            )
        images_by_file_name[file_name] = {
            **image_item,
            "id": coco_id,
            "file_name": file_name,
        }

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)

    return images_by_file_name, annotations_by_image


def resolve_selected_images(
    requested_names: List[str],
    image_dir: Path,
    images_by_file_name: Dict[str, dict],
) -> List[dict]:
    """Resolve images by exact file name and link them to their COCO entries."""
    selected: List[dict] = []
    errors: List[str] = []

    for file_name in requested_names:
        image_path = image_dir / file_name
        if not image_path.is_file():
            errors.append(f"Image file not found: {image_path}")
            continue

        image_item = images_by_file_name.get(file_name)
        if image_item is None:
            errors.append(
                f"Image {file_name} is present on disk but is not listed in the COCO file"
            )
            continue

        selected.append({
            "file_name": file_name,
            "coco_image_id": int(image_item["id"]),
            "image_path": image_path,
        })

    if errors:
        raise FileNotFoundError(
            "Some requested images could not be resolved:\n  - "
            + "\n  - ".join(errors)
        )

    print("\nSelected images:", flush=True)
    for item in selected:
        print(
            f"  {item['file_name']} -> COCO image_id={item['coco_image_id']}",
            flush=True,
        )

    return selected


# ============================================================
# UTILITY MASCHERE E VISUALIZZAZIONE
# ============================================================


def polygon_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    if not isinstance(segmentation, list):
        return mask

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)

    return mask


def mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32,
    )


def mask_to_centroid_point(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))

    if (
        0 <= cy < mask.shape[0]
        and 0 <= cx < mask.shape[1]
        and mask[cy, cx] > 0
    ):
        return np.array([cx, cy], dtype=np.float32)

    distances = (xs - cx) ** 2 + (ys - cy) ** 2
    closest_index = int(np.argmin(distances))
    return np.array(
        [xs[closest_index], ys[closest_index]],
        dtype=np.float32,
    )


def semantic_to_color(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, bgr in CLASS_COLORS.items():
        if class_id == 0:
            continue
        color_mask[mask == class_id] = bgr

    return color_mask


def make_overlay(
    image_bgr: np.ndarray,
    semantic_mask: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    color_mask = semantic_to_color(semantic_mask)
    overlay = image_bgr.copy()
    foreground = semantic_mask > 0

    if np.any(foreground):
        blended = cv2.addWeighted(
            image_bgr,
            1.0 - alpha,
            color_mask,
            alpha,
            0.0,
        )
        overlay[foreground] = blended[foreground]

    return color_mask, overlay


def add_image_label(
    image: np.ndarray,
    image_name: str,
    prompt_mode: str,
) -> np.ndarray:
    output = image.copy()
    text = f"{image_name} - MedSAM {prompt_mode.upper()}"

    cv2.rectangle(output, (0, 0), (950, 52), (0, 0, 0), thickness=-1)
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
    image_paths: List[Path],
    output_path: Path,
    columns: int = 3,
    thumb_width: int = 640,
) -> None:
    if not image_paths:
        return

    tiles: List[np.ndarray] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        scale = thumb_width / image.shape[1]
        thumb_height = max(1, int(round(image.shape[0] * scale)))
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

    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y0 = row * tile_height
        x0 = column * thumb_width
        height, width = tile.shape[:2]
        sheet[y0:y0 + height, x0:x0 + width] = tile

    cv2.imwrite(str(output_path), sheet)


# ============================================================
# MEDSAM
# ============================================================


@torch.no_grad()
def decode_mask(
    model,
    image_embedding: torch.Tensor,
    sparse_embeddings: torch.Tensor,
    dense_embeddings: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    prediction = torch.sigmoid(low_res_logits)
    prediction = torch.nn.functional.interpolate(
        prediction,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    prediction = prediction.squeeze().cpu().numpy()

    return (prediction > 0.5).astype(np.uint8)


@torch.no_grad()
def infer_bbox(
    model,
    image_embedding: torch.Tensor,
    box_1024: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    box_tensor = torch.as_tensor(
        box_1024,
        dtype=torch.float32,
        device=image_embedding.device,
    )[None, :]

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )

    return decode_mask(
        model,
        image_embedding,
        sparse_embeddings,
        dense_embeddings,
        height,
        width,
    )


@torch.no_grad()
def infer_point(
    model,
    image_embedding: torch.Tensor,
    point_1024: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    point_tensor = torch.as_tensor(
        point_1024,
        dtype=torch.float32,
        device=image_embedding.device,
    )[None, None, :]
    label_tensor = torch.ones(
        (1, 1),
        dtype=torch.int,
        device=image_embedding.device,
    )

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=(point_tensor, label_tensor),
        boxes=None,
        masks=None,
    )

    return decode_mask(
        model,
        image_embedding,
        sparse_embeddings,
        dense_embeddings,
        height,
        width,
    )


def prepare_image_embedding(model, image_bgr: np.ndarray, device: torch.device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_1024 = cv2.resize(
        image_rgb,
        (1024, 1024),
        interpolation=cv2.INTER_LINEAR,
    )
    image_1024 = image_1024.astype(np.float32)

    denominator = max(
        float(image_1024.max() - image_1024.min()),
        1e-8,
    )
    image_1024 = (
        image_1024 - image_1024.min()
    ) / denominator

    tensor = (
        torch.from_numpy(image_1024)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        return model.image_encoder(tensor)


# ============================================================
# ESECUZIONE DI UNA MODALITA' DI PROMPT
# ============================================================


def run_prompt_mode(
    prompt_mode: str,
    model,
    device: torch.device,
    op_name: str,
    fold_name: str | None,
    selected_images: List[dict],
    annotations_by_image: Dict[int, List[dict]],
    output_root: Path,
    alpha: float,
) -> None:
    mode_name = f"medsam_{prompt_mode}"
    output_dir = output_root / mode_name
    semantic_dir = output_dir / "semantic_masks"
    color_dir = output_dir / "color_masks"
    overlay_dir = output_dir / "overlays"

    for folder in (semantic_dir, color_dir, overlay_dir):
        folder.mkdir(parents=True, exist_ok=True)

    manifest_frames = []
    overlay_paths: List[Path] = []

    print("\n" + "=" * 90, flush=True)
    print(f"Modalita':   {prompt_mode}", flush=True)
    print(f"Paziente:    {op_name}", flush=True)
    if fold_name is not None:
        print(f"Fold:        {fold_name}", flush=True)
    print(f"Output:      {output_dir}", flush=True)
    print("=" * 90, flush=True)

    for position, selected in enumerate(selected_images, start=1):
        coco_image_id = int(selected["coco_image_id"])
        image_path = Path(selected["image_path"])

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Immagine non leggibile: {image_path}")

        height, width = image_bgr.shape[:2]
        image_embedding = prepare_image_embedding(model, image_bgr, device)
        semantic_mask = np.zeros((height, width), dtype=np.uint8)

        scale_x = 1024.0 / width
        scale_y = 1024.0 / height
        used_instances = 0
        skipped_lamina = 0
        processed_instances = []

        for annotation in annotations_by_image.get(coco_image_id, []):
            coco_category_id = int(annotation["category_id"])
            class_name = COCO_CLASS_NAMES.get(coco_category_id)

            if class_name == "lamina":
                skipped_lamina += 1
                continue
            if class_name is None:
                continue

            canonical_id = NAME_TO_CANONICAL_ID.get(class_name)
            if canonical_id is None or canonical_id == 0:
                continue

            ground_truth_instance = polygon_to_mask(
                annotation.get("segmentation", []),
                height,
                width,
            )
            if ground_truth_instance.sum() == 0:
                continue

            if prompt_mode == "bbox":
                prompt = mask_to_bbox(ground_truth_instance)
                if prompt is None:
                    continue

                prompt_1024 = prompt.copy()
                prompt_1024[[0, 2]] *= scale_x
                prompt_1024[[1, 3]] *= scale_y

                predicted_instance = infer_bbox(
                    model,
                    image_embedding,
                    prompt_1024,
                    height,
                    width,
                )

                prompt_manifest = [float(value) for value in prompt.tolist()]

            elif prompt_mode == "point":
                prompt = mask_to_centroid_point(ground_truth_instance)
                if prompt is None:
                    continue

                prompt_1024 = prompt.copy()
                prompt_1024[0] *= scale_x
                prompt_1024[1] *= scale_y

                predicted_instance = infer_point(
                    model,
                    image_embedding,
                    prompt_1024,
                    height,
                    width,
                )

                prompt_manifest = [float(value) for value in prompt.tolist()]

            else:
                raise ValueError(f"Prompt mode non valido: {prompt_mode}")

            # L'ultima istanza processata prevale nelle aree sovrapposte,
            # coerentemente con lo script originale.
            semantic_mask[predicted_instance == 1] = canonical_id
            used_instances += 1

            processed_instances.append({
                "annotation_id": annotation.get("id"),
                "coco_category_id": coco_category_id,
                "canonical_class_id": canonical_id,
                "class_name": class_name,
                "prompt": prompt_manifest,
                "predicted_pixels": int(predicted_instance.sum()),
            })

        color_mask, overlay = make_overlay(
            image_bgr,
            semantic_mask,
            alpha,
        )
        overlay = add_image_label(
            overlay,
            selected["file_name"],
            prompt_mode,
        )

        stem = image_path.stem
        semantic_path = semantic_dir / f"{stem}_mask.png"
        color_path = color_dir / f"{stem}_color_mask.png"
        overlay_path = overlay_dir / f"{stem}_overlay.png"

        if not cv2.imwrite(str(semantic_path), semantic_mask):
            raise RuntimeError(f"Errore nel salvataggio di {semantic_path}")
        if not cv2.imwrite(str(color_path), color_mask):
            raise RuntimeError(f"Errore nel salvataggio di {color_path}")
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Errore nel salvataggio di {overlay_path}")

        predicted_ids = sorted(
            int(value)
            for value in np.unique(semantic_mask)
            if int(value) != 0
        )
        predicted_classes = [
            CANONICAL_CLASS_NAMES[class_id]
            for class_id in predicted_ids
        ]

        overlay_paths.append(overlay_path)
        manifest_frames.append({
            "position": position,
            "input_file_name": selected["file_name"],
            "coco_image_id": coco_image_id,
            "file_name": selected["file_name"],
            "image_path": str(image_path),
            "semantic_mask": str(semantic_path),
            "color_mask": str(color_path),
            "overlay": str(overlay_path),
            "instances_processed": used_instances,
            "lamina_instances_ignored": skipped_lamina,
            "predicted_class_ids": predicted_ids,
            "predicted_classes": predicted_classes,
            "instances": processed_instances,
        })

        print(
            f"[{position:02d}/{len(selected_images)}] "
            f"{selected['file_name']}: "
            f"{used_instances} istanze processate, "
            f"{skipped_lamina} lamina ignorate",
            flush=True,
        )

        del image_embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "model": "medsam",
        "prompt_mode": prompt_mode,
        "patient": op_name,
        "fold": fold_name,
        "checkpoint": str(CHECKPOINT),
        "alpha": alpha,
        "canonical_classes": CANONICAL_CLASS_NAMES,
        "lamina_ignored": True,
        "images": manifest_frames,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    contact_sheet_path = (
        output_dir
        / f"{op_name}_medsam_{prompt_mode}_{len(selected_images)}_frames.jpg"
    )
    create_contact_sheet(
        overlay_paths,
        contact_sheet_path,
        columns=3,
    )

    print(f"\nMedSAM {prompt_mode} completato.", flush=True)
    print(f"Manifest:      {manifest_path}", flush=True)
    print(f"Contact sheet: {contact_sheet_path}", flush=True)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qualitative MedSAM inference on selected ExoDisc images. "
            "Images are selected by exact file name, while prompts are derived "
            "from the corresponding COCO annotations. The lamina class is ignored."
        )
    )
    parser.add_argument(
        "--op",
        required=True,
        choices=VALID_OPS,
        help="Operation to process.",
    )
    parser.add_argument(
        "--fold",
        choices=("fold1", "fold2", "fold3", "fold4"),
        default=None,
        help=(
            "Fold associated with the qualitative evaluation. MedSAM uses one "
            "shared checkpoint, so this value is recorded only in the manifest."
        ),
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        help=(
            "Exact image file names, separated by spaces or commas. Example: "
            "--images op1_0123.png op1_0456.png"
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        default="all",
        choices=("bbox", "point", "all"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity between 0 and 1.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device, for example cuda:0 or cpu.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")

    try:
        requested_names = parse_image_names(args.images)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    image_dir = DATASET_ROOT / args.op / "images"
    coco_path = (
        DATASET_ROOT
        / args.op
        / "annotations"
        / "instances_default.json"
    )
    output_root = QUALITATIVE_ROOT / args.op

    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {CHECKPOINT}")

    with coco_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    images_by_file_name, annotations_by_image = build_coco_indices(coco)
    selected_images = resolve_selected_images(
        requested_names=requested_names,
        image_dir=image_dir,
        images_by_file_name=images_by_file_name,
    )

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = requested_device

    print("\nConfiguration:", flush=True)
    print(f"  Operation:  {args.op}", flush=True)
    print(f"  Fold:       {args.fold}", flush=True)
    print(f"  Images:     {image_dir}", flush=True)
    print(f"  COCO:       {coco_path}", flush=True)
    print(f"  Checkpoint: {CHECKPOINT}", flush=True)
    print(f"  Output:     {output_root}", flush=True)
    print(f"  Device:     {device}", flush=True)

    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(device)}", flush=True)

    model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
    model = model.to(device)
    model.eval()

    prompt_modes = (
        ["bbox", "point"]
        if args.prompt_mode == "all"
        else [args.prompt_mode]
    )

    for prompt_mode in prompt_modes:
        run_prompt_mode(
            prompt_mode=prompt_mode,
            model=model,
            device=device,
            op_name=args.op,
            fold_name=args.fold,
            selected_images=selected_images,
            annotations_by_image=annotations_by_image,
            output_root=output_root,
            alpha=args.alpha,
        )

    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    main()