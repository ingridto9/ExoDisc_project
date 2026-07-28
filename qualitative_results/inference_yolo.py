#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURAZIONE GENERALE
# ============================================================

HOME = Path.home()
PROJECT_ROOT = HOME / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"
RESULTS_ROOT = HOME / "ExoDisc_results"

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

DEFAULT_COLOR = (200, 200, 200)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# UTILS
# ============================================================

def normalize_names(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def resolve_selected_images(
    requested_names: Iterable[str],
    images_dir: Path,
) -> List[Path]:
    """Resolve image files by exact filename.

    Each value passed through --images must match an existing filename inside
    the selected operation directory. Matching is exact and does not rely on
    COCO image IDs or numeric frame indices.
    """
    available = {
        path.name: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }

    resolved: List[Path] = []
    missing: List[str] = []

    for raw_name in requested_names:
        for token in raw_name.split(','):
            name = token.strip()
            if not name:
                continue

            image_path = available.get(name)
            if image_path is None:
                missing.append(name)
            else:
                resolved.append(image_path)

    if not resolved and not missing:
        raise argparse.ArgumentTypeError('No image filename was provided.')

    if missing:
        raise FileNotFoundError(
            'The following image files were not found in '
            f'{images_dir}:\n  - ' + '\n  - '.join(missing)
        )

    return list(dict.fromkeys(resolved))


def resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = mask.astype(np.uint8)
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0


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
            print(f"ATTENZIONE: impossibile leggere {image_path}", file=sys.stderr)
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
        row = index // columns
        column = index % columns
        height, width = tile.shape[:2]

        y0 = row * tile_height
        x0 = column * thumb_width
        sheet[y0:y0 + height, x0:x0 + width] = tile

    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Impossibile salvare la contact sheet: {output_path}")


# ============================================================
# INFERENZA
# ============================================================

def run_inference(args: argparse.Namespace) -> None:
    op = args.op
    fold = args.fold
    requested_names = args.images

    images_dir = DATASET_ROOT / op / "images"
    checkpoint = (
        RESULTS_ROOT
        / "yolo11"
        / fold
        / "weights"
        / "best.pt"
    )
    output_dir = (
        RESULTS_ROOT
        / "qualitative_inference_comparison"
        / op
        / "yolo11"
    )

    if not images_dir.exists():
        raise FileNotFoundError(f"Cartella immagini non trovata: {images_dir}")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint}")

    selected_images = resolve_selected_images(
        requested_names=requested_names,
        images_dir=images_dir,
    )

    masks_dir = output_dir / "semantic_masks"
    color_masks_dir = output_dir / "color_masks"
    overlays_dir = output_dir / "overlays"

    for directory in (masks_dir, color_masks_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("YOLO11 QUALITATIVE INFERENCE")
    print("=" * 72)
    print(f"Paziente:       {op}")
    print(f"Fold:           {fold}")
    print(f"Number of images: {len(selected_images)}")
    print(f"Checkpoint:     {checkpoint}")
    print(f"Immagini:       {images_dir}")
    print(f"Output:         {output_dir}")
    print("=" * 72)

    print("\nSelected images:")
    for image_path in selected_images:
        print(f"  {image_path.name}")

    model = YOLO(str(checkpoint))
    class_names = normalize_names(model.names)

    print("\nClassi YOLO:")
    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")

    manifest = []
    overlay_paths: List[Path] = []

    for position, image_path in enumerate(selected_images, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Impossibile leggere: {image_path}")

        height, width = image_bgr.shape[:2]

        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )

        result = results[0]
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        color_mask = np.zeros_like(image_bgr)
        detections = []

        if result.masks is not None and result.boxes is not None:
            instance_masks = result.masks.data.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()

            # Prima le predizioni meno sicure; quelle piu' sicure restano sopra.
            order = np.argsort(confidences)

            for instance_index in order:
                class_id = int(class_ids[instance_index])
                confidence = float(confidences[instance_index])
                class_name = class_names.get(class_id, f"class_{class_id}")
                normalized_class_name = class_name.strip().lower()

                # Lamina viene sempre ignorata.
                if normalized_class_name == "lamina":
                    continue

                binary_mask = resize_binary_mask(
                    instance_masks[instance_index],
                    width=width,
                    height=height,
                )

                # 0 rimane background; class_id + 1 rappresenta la classe YOLO.
                semantic_value = class_id + 1
                semantic_mask[binary_mask] = semantic_value

                color = CLASS_COLORS.get(normalized_class_name, DEFAULT_COLOR)
                color_mask[binary_mask] = color

                detections.append(
                    {
                        "class_id": class_id,
                        "semantic_value": semantic_value,
                        "class_name": class_name,
                        "confidence": confidence,
                        "pixels": int(binary_mask.sum()),
                    }
                )

        overlay = cv2.addWeighted(
            image_bgr,
            1.0,
            color_mask,
            args.alpha,
            0.0,
        )

        label = f"{op} - {image_path.name} - YOLO11"
        cv2.rectangle(overlay, (0, 0), (760, 52), (0, 0, 0), thickness=-1)
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

        stem = image_path.stem
        semantic_path = masks_dir / f"{stem}_mask.png"
        color_path = color_masks_dir / f"{stem}_color_mask.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"

        if not cv2.imwrite(str(semantic_path), semantic_mask):
            raise RuntimeError(f"Impossibile salvare: {semantic_path}")
        if not cv2.imwrite(str(color_path), color_mask):
            raise RuntimeError(f"Impossibile salvare: {color_path}")
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Impossibile salvare: {overlay_path}")

        overlay_paths.append(overlay_path)
        manifest.append(
            {
                "position": position,
                "file_name": image_path.name,
                "image_path": str(image_path),
                "semantic_mask": str(semantic_path),
                "color_mask": str(color_path),
                "overlay": str(overlay_path),
                "detections": detections,
            }
        )

        print(
            f"[{position:02d}/{len(selected_images):02d}] "
            f"{image_path.name}: {len(detections)} istanze"
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": "yolo11",
                "patient": op,
                "fold": fold,
                "requested_images": requested_names,
                "checkpoint": str(checkpoint),
                "images_dir": str(images_dir),
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "device": str(args.device),
                "alpha": args.alpha,
                "frames": manifest,
            },
            file,
            indent=2,
        )

    contact_sheet_path = output_dir / f"{op}_yolo11_{len(selected_images)}_frames.jpg"
    create_contact_sheet(
        overlay_paths,
        contact_sheet_path,
        columns=args.contact_columns,
        thumb_width=args.thumb_width,
    )

    print("\nInference completed.")
    print(f"Manifest:      {manifest_path}")
    print(f"Contact sheet: {contact_sheet_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO11 inference on selected image files from op1, op2, op3, or op4. "
            "Images are selected by exact filename."
        )
    )

    parser.add_argument(
        "--op",
        required=True,
        choices=["op1", "op2", "op3", "op4"],
        help="Paziente/operazione da processare.",
    )
    parser.add_argument(
        "--fold",
        required=True,
        choices=["fold1", "fold2", "fold3", "fold4"],
        help="Fold dal quale caricare il checkpoint YOLO11.",
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        help=(
            "Exact image filenames. Examples: --images op1_0123.png op1_0456.png "
            "or --images op1_0123.png,op1_0456.png"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--contact-columns", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=640)

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha deve essere compreso tra 0 e 1.")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf deve essere compreso tra 0 e 1.")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou deve essere compreso tra 0 e 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz deve essere positivo.")
    if args.contact_columns <= 0:
        parser.error("--contact-columns deve essere positivo.")
    if args.thumb_width <= 0:
        parser.error("--thumb-width deve essere positivo.")

    run_inference(args)


if __name__ == "__main__":
    main()