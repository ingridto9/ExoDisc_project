# Mask2Former benchmark

This directory contains the code used to train and evaluate the Hugging Face implementation of Mask2Former on the ExoDisc patient-wise four-fold benchmark.

## Expected directory structure

The repository and the ExoDisc dataset must be arranged as follows by default:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   ├── op2/
│   ├── op3/
│   └── op4/
└── mask2former/
    ├── train_mask2former.py
    └── requirements-mask2former_medsam_dinov2-lock.txt
```

Custom dataset and output paths can also be provided through command-line arguments.

---

## Software environment

`requirements-mask2former_medsam_dinov2-lock.txt` documents the software environment used for the experiments reported in the paper.

The original environment was based on

```text
pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
```

To reproduce the original software environment without using the container:

```bash
python -m venv .venv-mask2former
source .venv-mask2former/bin/activate
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

---

## Pretrained checkpoint

The benchmark uses the official Hugging Face checkpoint

```text
facebook/mask2former-swin-tiny-ade-semantic
```

The checkpoint is downloaded automatically the first time the script is executed.

If desired, another Hugging Face checkpoint can be specified through

```bash
--checkpoint <checkpoint_name>
```

---

## Cross-validation folds

The benchmark uses patient-wise leave-one-operation-out cross-validation:

| Fold | Training operations | Validation operation |
|---|---|---|
| fold1 | op2, op3, op4 | op1 |
| fold2 | op1, op3, op4 | op2 |
| fold3 | op1, op2, op4 | op3 |
| fold4 | op1, op2, op3 | op4 |

---

## Training

Train all four folds:

```bash
python mask2former/train_mask2former.py
```

Train a single fold:

```bash
python mask2former/train_mask2former.py \
    --folds fold1
```

The default training configuration used in the paper is:

| Parameter | Value |
|---|---:|
| Checkpoint | `facebook/mask2former-swin-tiny-ade-semantic` |
| Epochs | 30 |
| Image size | 512 |
| Batch size | 1 |
| Learning rate | 5e-5 |

---

## Output

Training results are written to

```text
~/ExoDisc_results/
└── semantic_benchmark/
    └── hf_mask2former/
```

Each fold generates

```text
best.pt
last.pt
config.json
classes.txt
results.csv
```
