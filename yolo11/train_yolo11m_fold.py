#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import subprocess

import yaml


FOLDS = {
    "fold1": {"train": ["op2", "op3", "op4"], "val": ["op1"]},
    "fold2": {"train": ["op1", "op3", "op4"], "val": ["op2"]},
    "fold3": {"train": ["op1", "op2", "op4"], "val": ["op3"]},
    "fold4": {"train": ["op1", "op2", "op3"], "val": ["op4"]},
}

NAMES = {
    0: "aspirator",
    1: "burr",
    2: "retractor",
    3: "spatula",
    4: "forceps",
    5: "scalpel",
    6: "curettes",
    7: "electrocautery",
    8: "disc",
    9: "dura",
    10: "ligament",
    11: "herniation",
}

# Mapping from the original ExoDisc YOLO class IDs to the 12 foreground
# benchmark classes. Class 3 (lamina) is excluded from the benchmark.
CLASS_MAP = {
    4: 0,    # aspirator
    5: 1,    # burr
    6: 2,    # retractor
    7: 3,    # spatula
    8: 4,    # forceps
    9: 5,    # scalpel
    10: 6,   # curettes
    11: 7,   # electrocautery
    0: 8,    # disc
    1: 9,    # dura
    2: 10,   # ligament
    12: 11,  # herniation
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_root = Path.home() / "ExoDisc_project"

    parser = argparse.ArgumentParser(
        description="Prepare one patient-wise ExoDisc fold and train YOLO11m-seg."
    )
    parser.add_argument(
        "--fold",
        required=True,
        choices=sorted(FOLDS),
        help="Patient-wise cross-validation fold to train.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "ExoDisc",
        help="ExoDisc dataset root containing op1, op2, op3 and op4.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Directory used to store the generated YOLO fold datasets.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path.home() / "ExoDisc_results",
        help="Root directory for Ultralytics training outputs.",
    )
    parser.add_argument("--model", default="yolo11m-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def remap_and_copy_label(source: Path, destination: Path) -> tuple[bool, int, int]:
    output_lines = []
    skipped_lamina = 0
    skipped_unknown = 0

    with source.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            old_class_id = int(parts[0])

            if old_class_id == 3:
                skipped_lamina += 1
                continue

            if old_class_id not in CLASS_MAP:
                skipped_unknown += 1
                continue

            new_class_id = CLASS_MAP[old_class_id]
            output_lines.append(" ".join([str(new_class_id)] + parts[1:]))

    if not output_lines:
        return False, skipped_lamina, skipped_unknown

    with destination.open("w", encoding="utf-8") as file:
        file.write("\n".join(output_lines) + "\n")

    return True, skipped_lamina, skipped_unknown


def collect_images_and_labels(
    dataset_root: Path,
    operations: list[str],
    split_name: str,
    images_out: Path,
    labels_out: Path,
) -> None:
    total_images = 0
    missing_labels = 0
    empty_after_filter = 0
    total_lamina_instances_skipped = 0
    total_unknown_instances_skipped = 0

    for operation in operations:
        images_dir = dataset_root / operation / "images"
        labels_dir = dataset_root / operation / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"YOLO labels directory not found: {labels_dir}")

        image_files = sorted(
            path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
        )

        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels += 1
                continue

            image_destination = images_out / f"{operation}_{image_path.name}"
            label_destination = labels_out / f"{operation}_{image_path.stem}.txt"

            label_ok, skipped_lamina, skipped_unknown = remap_and_copy_label(
                label_path, label_destination
            )
            total_lamina_instances_skipped += skipped_lamina
            total_unknown_instances_skipped += skipped_unknown

            if not label_ok:
                empty_after_filter += 1
                continue

            symlink_file(image_path, image_destination)
            total_images += 1

    print(f"{split_name}: images used = {total_images}")
    print(f"{split_name}: images excluded because labels were missing = {missing_labels}")
    print(
        f"{split_name}: images excluded after lamina removal = {empty_after_filter}"
    )
    print(
        f"{split_name}: lamina instances removed = "
        f"{total_lamina_instances_skipped}"
    )
    print(
        f"{split_name}: unknown-class instances removed = "
        f"{total_unknown_instances_skipped}"
    )


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    fold_config = FOLDS[args.fold]
    experiment_dir = project_root / "experiments" / "yolo" / args.fold

    images_train_dir = experiment_dir / "images" / "train"
    images_val_dir = experiment_dir / "images" / "val"
    labels_train_dir = experiment_dir / "labels" / "train"
    labels_val_dir = experiment_dir / "labels" / "val"

    reset_dir(experiment_dir)
    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    collect_images_and_labels(
        dataset_root,
        fold_config["train"],
        "TRAIN",
        images_train_dir,
        labels_train_dir,
    )
    collect_images_and_labels(
        dataset_root,
        fold_config["val"],
        "VAL",
        images_val_dir,
        labels_val_dir,
    )

    data_yaml = {
        "path": str(experiment_dir),
        "train": "images/train",
        "val": "images/val",
        "names": NAMES,
    }
    data_yaml_path = experiment_dir / "data.yaml"
    with data_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False)

    output_project = results_root / "yolo11"
    output_project.mkdir(parents=True, exist_ok=True)

    command = [
        "yolo",
        "segment",
        "train",
        f"model={args.model}",
        f"data={data_yaml_path}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"project={output_project}",
        f"name={args.fold}",
        f"patience={args.patience}",
        "plots=True",
        "save=True",
        "exist_ok=True",
    ]

    print(f"Generated fold dataset: {experiment_dir}")
    print("YOLO command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
