#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry


PROJECT_ROOT = Path.home() / "ExoDisc_project"
DATASET_ROOT = PROJECT_ROOT / "ExoDisc"
RESULTS_ROOT = (
    Path.home()
    / "ExoDisc_results"
    / "semantic_benchmark"
    / "medsam_prompting"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "medsam"
    / "medsam_vit_b.pth"
)


FOLDS = {
    "fold1": ["op1"],
    "fold2": ["op2"],
    "fold3": ["op3"],
    "fold4": ["op4"],
}


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_to_centroid_point(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))

    if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
        return np.array([cx, cy], dtype=np.float32)

    d = (xs - cx) ** 2 + (ys - cy) ** 2
    i = np.argmin(d)
    return np.array([xs[i], ys[i]], dtype=np.float32)


def polygon_to_mask(segmentation, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    if not isinstance(segmentation, list):
        return mask

    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)

    return mask


@torch.no_grad()
def medsam_inference_box(model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(
        box_1024,
        dtype=torch.float32,
        device=img_embed.device,
    )[None, :]

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    return decode_mask(model, img_embed, sparse_embeddings, dense_embeddings, H, W)


@torch.no_grad()
def medsam_inference_point(model, img_embed, point_1024, H, W):
    point_torch = torch.as_tensor(
        point_1024,
        dtype=torch.float32,
        device=img_embed.device,
    )[None, None, :]

    label_torch = torch.ones(
        (1, 1),
        dtype=torch.int,
        device=img_embed.device,
    )

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=(point_torch, label_torch),
        boxes=None,
        masks=None,
    )

    return decode_mask(model, img_embed, sparse_embeddings, dense_embeddings, H, W)


@torch.no_grad()
def decode_mask(model, img_embed, sparse_embeddings, dense_embeddings, H, W):
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)

    low_res_pred = torch.nn.functional.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    pred = low_res_pred.squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.uint8)


def compute_metrics(gt_semantic, pred_semantic, class_ids, cat_id_to_name):
    rows = []

    for cid in class_ids:
        gt = gt_semantic == cid
        pred = pred_semantic == cid

        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        dice_den = gt.sum() + pred.sum()

        if gt.sum() == 0 and pred.sum() == 0:
            iou = np.nan
            dice = np.nan
        else:
            iou = inter / union if union > 0 else 0.0
            dice = 2 * inter / dice_den if dice_den > 0 else 0.0

        rows.append({
            "class_id": cid,
            "class_name": cat_id_to_name.get(cid, str(cid)),
            "intersection": int(inter),
            "union": int(union),
            "gt_pixels": int(gt.sum()),
            "pred_pixels": int(pred.sum()),
            "iou": iou,
            "dice": dice,
        })

    return rows


def process_one_image(image_path, anns, model, device, class_ids, cat_id_to_name, prompt_mode):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Immagine non trovata: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    img_1024 = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_1024 = img_1024.astype(np.float32)
    img_1024 = (img_1024 - img_1024.min()) / max(
        img_1024.max() - img_1024.min(), 1e-8
    )

    img_1024_tensor = (
        torch.tensor(img_1024)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        img_embed = model.image_encoder(img_1024_tensor)

    scale_x = 1024 / W
    scale_y = 1024 / H

    gt_semantic = np.zeros((H, W), dtype=np.uint8)
    pred_semantic = np.zeros((H, W), dtype=np.uint8)

    instance_rows = []

    for idx, ann in enumerate(anns):
        class_id = int(ann["category_id"])
        ann_id = int(ann.get("id", idx))

        if class_id not in class_ids:
            continue

        gt_instance = polygon_to_mask(ann.get("segmentation", []), H, W)
        if gt_instance.sum() == 0:
            continue

        if prompt_mode == "bbox":
            bbox = mask_to_bbox(gt_instance)
            if bbox is None:
                continue

            prompt_1024 = bbox.copy()
            prompt_1024[[0, 2]] *= scale_x
            prompt_1024[[1, 3]] *= scale_y

            pred_instance = medsam_inference_box(
                model=model,
                img_embed=img_embed,
                box_1024=prompt_1024,
                H=H,
                W=W,
            )

        elif prompt_mode == "point":
            point = mask_to_centroid_point(gt_instance)
            if point is None:
                continue

            prompt_1024 = point.copy()
            prompt_1024[0] *= scale_x
            prompt_1024[1] *= scale_y

            pred_instance = medsam_inference_point(
                model=model,
                img_embed=img_embed,
                point_1024=prompt_1024,
                H=H,
                W=W,
            )

        else:
            raise ValueError(f"Prompt mode non valido: {prompt_mode}")

        gt_semantic[gt_instance == 1] = class_id
        pred_semantic[pred_instance == 1] = class_id

        inter = np.logical_and(gt_instance == 1, pred_instance == 1).sum()
        union = np.logical_or(gt_instance == 1, pred_instance == 1).sum()
        dice_den = gt_instance.sum() + pred_instance.sum()

        instance_rows.append({
            "ann_id": ann_id,
            "class_id": class_id,
            "class_name": cat_id_to_name.get(class_id, str(class_id)),
            "prompt_mode": prompt_mode,
            "iou": inter / union if union > 0 else 0.0,
            "dice": 2 * inter / dice_den if dice_den > 0 else 0.0,
            "gt_pixels": int(gt_instance.sum()),
            "pred_pixels": int(pred_instance.sum()),
        })

    image_class_rows = compute_metrics(
        gt_semantic,
        pred_semantic,
        class_ids,
        cat_id_to_name,
    )

    for r in image_class_rows:
        r["prompt_mode"] = prompt_mode

    return gt_semantic, pred_semantic, image_class_rows, instance_rows


def evaluate_fold(fold, dataset_root, out_root, model, device, prompt_mode, save_masks):
    fold_out = out_root / prompt_mode / fold
    fold_out.mkdir(parents=True, exist_ok=True)

    all_image_class_rows = []
    all_instance_rows = []

    for op in FOLDS[fold]:
        op_dir = dataset_root / op
        img_dir = op_dir / "images"
        ann_path = op_dir / "annotations" / "instances_default.json"

        with open(ann_path, "r") as f:
            coco = json.load(f)

        cat_id_to_name = {
            int(c["id"]): c.get("name", str(c["id"]))
            for c in coco.get("categories", [])
        }

        class_ids = sorted(cat_id_to_name.keys())

        anns_by_img = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for im in tqdm(coco["images"], desc=f"{prompt_mode} {fold} {op}"):
            image_name = Path(im["file_name"]).name
            image_path = img_dir / image_name
            anns = anns_by_img.get(im["id"], [])

            if len(anns) == 0:
                continue

            gt_sem, pred_sem, image_class_rows, instance_rows = process_one_image(
                image_path=image_path,
                anns=anns,
                model=model,
                device=device,
                class_ids=class_ids,
                cat_id_to_name=cat_id_to_name,
                prompt_mode=prompt_mode,
            )

            for r in image_class_rows:
                r["fold"] = fold
                r["op"] = op
                r["image"] = image_name
                all_image_class_rows.append(r)

            for r in instance_rows:
                r["fold"] = fold
                r["op"] = op
                r["image"] = image_name
                all_instance_rows.append(r)

            if save_masks:
                mask_dir = fold_out / "pred_semantic_masks" / op
                mask_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_dir / image_name), pred_sem)

    image_class_df = pd.DataFrame(all_image_class_rows)
    instance_df = pd.DataFrame(all_instance_rows)

    image_class_df.to_csv(fold_out / f"{fold}_image_class_metrics.csv", index=False)
    instance_df.to_csv(fold_out / f"{fold}_instance_metrics.csv", index=False)

    per_class = (
        image_class_df
        .groupby(["class_id", "class_name"], as_index=False)
        .agg(
            iou=("iou", "mean"),
            dice=("dice", "mean"),
            gt_pixels=("gt_pixels", "sum"),
            pred_pixels=("pred_pixels", "sum"),
        )
    )

    per_class.to_csv(fold_out / f"{fold}_per_class.csv", index=False)

    fold_metrics = {
        "prompt_mode": prompt_mode,
        "fold": fold,
        "val_ops": ",".join(FOLDS[fold]),
        "mIoU": float(per_class["iou"].mean()),
        "Dice": float(per_class["dice"].mean()),
        "n_images": int(image_class_df["image"].nunique()),
        "n_instances": int(len(instance_df)),
    }

    pd.DataFrame([fold_metrics]).to_csv(fold_out / f"{fold}_metrics.csv", index=False)

    return fold_metrics, per_class


def run_prompt_mode(prompt_mode, folds, dataset_root, out_root, model, device, save_masks):
    mode_out = out_root / prompt_mode
    mode_out.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []
    all_per_class = []

    for fold in folds:
        fold_metrics, per_class = evaluate_fold(
            fold=fold,
            dataset_root=dataset_root,
            out_root=out_root,
            model=model,
            device=device,
            prompt_mode=prompt_mode,
            save_masks=save_masks,
        )

        all_fold_metrics.append(fold_metrics)

        per_class["fold"] = fold
        per_class["prompt_mode"] = prompt_mode
        all_per_class.append(per_class)

    summary_folds = pd.DataFrame(all_fold_metrics)
    summary_folds.to_csv(mode_out / "summary_folds.csv", index=False)

    all_per_class_df = pd.concat(all_per_class, ignore_index=True)
    all_per_class_df.to_csv(mode_out / "summary_per_class_all_folds.csv", index=False)

    summary = pd.DataFrame([{
        "model": f"MedSAM_GT_{prompt_mode}_prompt",
        "prompt_mode": prompt_mode,
        "mIoU_mean": summary_folds["mIoU"].mean(),
        "mIoU_std": summary_folds["mIoU"].std(),
        "Dice_mean": summary_folds["Dice"].mean(),
        "Dice_std": summary_folds["Dice"].std(),
    }])

    summary.to_csv(mode_out / "summary_mean_std.csv", index=False)

    return summary_folds, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root containing op1, op2, op3, op4 and checkpoints.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=RESULTS_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--fold",
        default="all",
        choices=["all", "fold1", "fold2", "fold3", "fold4"],
    )
    parser.add_argument(
        "--prompt_modes",
        nargs="+",
        default=["bbox", "point"],
        choices=["bbox", "point"],
    )
    parser.add_argument("--save_masks", action="store_true")

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")

    for op in ["op1", "op2", "op3", "op4"]:
        op_dir = dataset_root / op
        image_dir = op_dir / "images"
        annotation_file = op_dir / "annotations" / "instances_default.json"
        if not image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
        if not annotation_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")

    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Dataset root:", dataset_root)
    print("Output root:", out_root)
    print("Checkpoint:", checkpoint)
    print("Prompt modes:", args.prompt_modes)

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()

    folds = ["fold1", "fold2", "fold3", "fold4"] if args.fold == "all" else [args.fold]

    all_mode_summaries = []

    for prompt_mode in args.prompt_modes:
        print(f"\n===== Running prompt mode: {prompt_mode} =====")

        summary_folds, summary = run_prompt_mode(
            prompt_mode=prompt_mode,
            folds=folds,
            dataset_root=dataset_root,
            out_root=out_root,
            model=model,
            device=device,
            save_masks=args.save_masks,
        )

        all_mode_summaries.append(summary)

        print(summary_folds)

    final_summary = pd.concat(all_mode_summaries, ignore_index=True)
    final_summary.to_csv(out_root / "summary_mean_std_all_prompt_modes.csv", index=False)

    print("\nDONE")
    print(final_summary)
    print("\nSaved in:", out_root)


if __name__ == "__main__":
    main()