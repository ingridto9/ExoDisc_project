# DINOv2 benchmark

This directory contains the code used to train and evaluate a DINOv2-based semantic segmentation model on the ExoDisc patient-wise four-fold benchmark.

The benchmark uses the official DINOv2 Base backbone (`facebook/dinov2-base`) with a lightweight semantic segmentation decoder trained on the ExoDisc dataset.

## Expected directory structure

The repository and the ExoDisc dataset must be arranged as follows by default:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   ├── op2/
│   ├── op3/
│   └── op4/
└── dinov2/
    ├── train_dinov2.py
    └── requirements-mask2former_medsam_dinov2-lock.txt
```

The scripts also accept custom dataset and results paths through command-line arguments.

## Software environment

`requirements-mask2former_medsam_dinov2-lock.txt` documents the software environment used for the experiments reported in the paper.

The original environment was based on:

```text
pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
```

To reproduce the original software environment without using the container:

```bash
python -m venv .venv-dinov2
source .venv-dinov2/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build used in the experiments:

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining packages:

```bash
pip install \
    numpy==1.26.4 \
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
    SimpleITK
```

A CUDA-capable NVIDIA GPU is recommended.

## Pretrained backbone

The benchmark uses the official Hugging Face checkpoint

```text
facebook/dinov2-base
```

The backbone is downloaded automatically the first time the script is executed.

A different Hugging Face checkpoint can be specified through

```bash
--model_name <checkpoint_name>
```

## Cross-validation folds

The benchmark uses patient-wise leave-one-operation-out cross-validation:

| Fold  | Training operations | Validation operation |
| ----- | ------------------- | -------------------- |
| fold1 | op2, op3, op4       | op1                  |
| fold2 | op1, op3, op4       | op2                  |
| fold3 | op1, op2, op4       | op3                  |
| fold4 | op1, op2, op3       | op4                  |

## Training

Train one fold from the repository root:

```bash
python dinov2/train_dinov2.py --fold fold1
```

Repeat the command for `fold2`, `fold3`, and `fold4`.

The default training configuration used in the paper is:

| Parameter              |                  Value |
| ---------------------- | ---------------------: |
| Backbone               | `facebook/dinov2-base` |
| Epochs                 |                     30 |
| Image size             |                    518 |
| Batch size             |                      2 |
| Backbone learning rate |                   1e-5 |
| Decoder learning rate  |                   1e-4 |
| Weight decay           |                   1e-4 |

The principal options can be overridden, for example:

```bash
python dinov2/train_dinov2.py \
    --fold fold1 \
    --dataset_root ~/ExoDisc_project/ExoDisc \
    --results_root ~/ExoDisc_results/semantic_benchmark/dinov2
```

## Output

Training results are written to:

```text
~/ExoDisc_results/
└── semantic_benchmark/
    └── dinov2/
```

Each fold generates:

```text
best_model.pt
last_model.pt
classes.txt
config.json
results.csv
per_class_best.csv
```

The training log (`results.csv`) reports the training and validation losses together with the corresponding mIoU and Dice scores for each epoch. The best-performing model is selected according to the validation mIoU.
