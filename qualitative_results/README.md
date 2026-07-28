# Qualitative results

This directory contains the scripts used to reproduce the qualitative comparison figures presented in the ExoDisc paper.

Each script loads the trained checkpoint of one or more benchmark models, performs inference on user-selected images from the ExoDisc dataset, and generates semantic segmentation masks, color-coded masks, and overlay visualizations.

## Expected directory structure

The repository, dataset, checkpoints, and trained models must be arranged as follows:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   ├── op2/
│   ├── op3/
│   └── op4/
│
├── checkpoints/
│   └── medsam/
│       └── medsam_vit_b.pth
│
├── qualitative_results/
│   ├── inference_yolo11.py
│   ├── inference_semantic_models.py
│   ├── inference_foundation_models.py
│   └── inference_medsam.py
│
└── ExoDisc_results/
    ├── yolo11/
    └── semantic_benchmark/
```

The inference scripts assume that the corresponding benchmark models have already been trained (except MedSAM, which uses the official pretrained checkpoint).

## Available scripts

| Script                           | Models                            |
| -------------------------------- | --------------------------------- |
| `inference_yolo11.py`            | YOLO11m-seg                       |
| `inference_semantic_models.py`   | U-Net++, DeepLabV3+, SegFormer-B2 |
| `inference_foundation_models.py` | DINOv2, Mask2Former               |
| `inference_medsam.py`            | MedSAM                            |

## Image selection

The qualitative inference scripts identify images by their **file names**.

For example:

```text
op1_0123.png
op2_0456.png
op3_0789.png
```

## Generated outputs

For every selected image, each script generates:

* semantic segmentation mask;
* color-coded semantic mask;
* RGB overlay;
* inference manifest (`manifest.json`).

The generated files preserve the original image name. For example,

```text
op1_0123.png
```

produces

```text
op1_0123_mask.png
op1_0123_color_mask.png
op1_0123_overlay.png
```

Each script also generates a contact sheet summarizing all processed images.

## Output directory

By default, all qualitative results are written to

```text
~/ExoDisc_results/
└── qualitative_inference_comparison/
```

with one subdirectory for each operation and model.

## Purpose

These scripts were used to generate the qualitative comparison figures presented in the ExoDisc paper and can be used to visualize predictions on arbitrary images from the released dataset.
