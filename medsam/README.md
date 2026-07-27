# MedSAM prompting benchmark

This directory contains the evaluation pipeline used to reproduce the MedSAM prompting experiments reported in the ExoDisc paper.

MedSAM is evaluated using ground-truth instance annotations under two prompting strategies:

* bounding-box prompts;
* single positive point prompts.

No model training or fine-tuning is performed.

## Expected directory structure

The repository, ExoDisc dataset, and MedSAM checkpoint must be arranged as follows by default:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   │   ├── images/
│   │   └── annotations/
│   │       └── instances_default.json
│   ├── op2/
│   ├── op3/
│   └── op4/
├── checkpoints/
│   └── medsam/
│       └── medsam_vit_b.pth
└── medsam/
    ├── evaluate_medsam_prompting.py
    └── requirements-mask2former-medsam-dinov2-lock.txt
```

Custom dataset, checkpoint, and output paths can also be provided through command-line arguments.

## Software environment

`requirements-mask2former-medsam-dinov2-lock.txt` documents the software environment used for the experiments reported in the paper. The environment was shared with the Mask2Former and DINOv2 pipelines and therefore includes packages that are not directly required by this MedSAM evaluation script.

The original container was based on:

```text
pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
```

To reproduce the original environment without using the container, create and activate a Python environment:

```bash
python -m venv .venv-medsam
source .venv-medsam/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build used in the original experiments:

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies listed in the lock file:

```bash
pip install numpy==1.26.4 \
  transformers==4.49.0 \
  accelerate \
  safetensors \
  huggingface_hub \
  opencv-python-headless \
  pillow \
  pandas \
  matplotlib \
  tqdm \
  scikit-learn \
  pyyaml \
  scipy \
  monai \
  scikit-image \
  SimpleITK \
  segment-anything
```

Finally, install MedSAM from the official repository:

```bash
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM
pip install -e .
cd ..
```

A CUDA-capable NVIDIA GPU is recommended. Users with a different CUDA configuration must install the corresponding compatible PyTorch build.

## MedSAM checkpoint

The pretrained MedSAM ViT-B checkpoint is not distributed with this repository.

Download the official [MedSAM ViT-B checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) and place the file at:

```text
~/ExoDisc_project/checkpoints/medsam/medsam_vit_b.pth
```

A different checkpoint location can be provided with:

```bash
--checkpoint /path/to/medsam_vit_b.pth
```

## Evaluation folds

The evaluation follows the same leave-one-operation-out organization used throughout the ExoDisc benchmark:

| Fold  | Evaluated operation |
| ----- | ------------------- |
| fold1 | op1                 |
| fold2 | op2                 |
| fold3 | op3                 |
| fold4 | op4                 |

Since MedSAM is evaluated in a prompting setting without training, each fold identifies only the operation used for evaluation.

## Prompt generation

Prompts are generated automatically from the released ground-truth instance annotations.

For bounding-box prompting, the tight bounding box enclosing each ground-truth instance is used.

For point prompting, one positive point is placed at the centroid of the ground-truth instance. If the centroid does not fall inside the instance mask, the nearest foreground pixel is used.

Predictions are generated independently for each annotated instance and then combined into semantic masks for evaluation.

## Evaluation

Run the script from the repository root.

Evaluate all four folds using both prompting strategies:

```bash
python medsam/evaluate_medsam_prompting.py
```

Evaluate only bounding-box prompting:

```bash
python medsam/evaluate_medsam_prompting.py \
  --prompt_modes bbox
```

Evaluate only point prompting:

```bash
python medsam/evaluate_medsam_prompting.py \
  --prompt_modes point
```

Evaluate a single fold:

```bash
python medsam/evaluate_medsam_prompting.py \
  --fold fold1
```

Use custom paths:

```bash
python medsam/evaluate_medsam_prompting.py \
  --dataset_root ~/ExoDisc_project/ExoDisc \
  --checkpoint ~/ExoDisc_project/checkpoints/medsam/medsam_vit_b.pth \
  --out_root ~/ExoDisc_results/semantic_benchmark/medsam_prompting
```

Save the predicted semantic masks in addition to the numerical results:

```bash
python medsam/evaluate_medsam_prompting.py \
  --save_masks
```

## Output

By default, results are written to:

```text
~/ExoDisc_results/
└── semantic_benchmark/
    └── medsam_prompting/
        ├── bbox/
        └── point/
```

For each prompting strategy and fold, the script generates:

```text
<fold>_image_class_metrics.csv
<fold>_instance_metrics.csv
<fold>_per_class.csv
<fold>_metrics.csv
```

For each prompting strategy, it also generates:

```text
summary_folds.csv
summary_per_class_all_folds.csv
summary_mean_std.csv
```

When both prompting strategies are evaluated, the combined summary is saved as:

```text
summary_mean_std_all_prompt_modes.csv
```

Predicted semantic masks are generated only when the `--save_masks` option is used.
