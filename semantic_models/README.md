# Semantic models benchmark

This directory contains the code used to train and evaluate three semantic segmentation models on the ExoDisc patient-wise four-fold benchmark:

* U-Net++
* DeepLabV3+
* SegFormer-B2

All models are implemented using the `segmentation_models_pytorch` library and share the same training pipeline, evaluation protocol, and software environment.

## Expected directory structure

The repository and the ExoDisc dataset must be arranged as follows by default:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   ├── op2/
│   ├── op3/
│   └── op4/
└── semantic_models/
    ├── train_semantic_models.py
    └── requirements-semantic-models-lock.txt
```

The scripts also accept custom dataset and results paths through command-line arguments.

## Software environment

`requirements-semantic-models-lock.txt` documents the software environment used for the experiments reported in the paper.

The original environment was based on:

```text
ultralytics/ultralytics:latest
```

To reproduce the original software environment without using the container:

```bash
python -m venv .venv-semantic
source .venv-semantic/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build used in the experiments:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining packages:

```bash
pip install \
    numpy==2.4.3 \
    segmentation_models_pytorch==0.5.0 \
    timm==1.0.27 \
    opencv-python-headless==5.0.0.93 \
    pyyaml==6.0.3 \
    pandas==3.0.3 \
    matplotlib==3.11.0 \
    pillow==12.1.1 \
    tqdm==4.68.4 \
    scikit-learn==1.9.0 \
    scipy==1.18.0 \
    huggingface_hub==1.23.0 \
    safetensors==0.8.0
```

A CUDA-capable NVIDIA GPU is recommended. The commands above reproduce the original CUDA 12.8 environment. Users with a different CUDA configuration must install a compatible PyTorch build.

## Cross-validation folds

The benchmark uses patient-wise leave-one-operation-out cross-validation:

| Fold  | Training operations | Validation operation |
| ----- | ------------------- | -------------------- |
| fold1 | op2, op3, op4       | op1                  |
| fold2 | op1, op3, op4       | op2                  |
| fold3 | op1, op2, op4       | op3                  |
| fold4 | op1, op2, op3       | op4                  |

## Training

Train all three models on all four folds:

```bash
python semantic_models/train_semantic_models.py
```

Train only U-Net++:

```bash
python semantic_models/train_semantic_models.py \
    --models unetpp
```

Train only DeepLabV3+:

```bash
python semantic_models/train_semantic_models.py \
    --models deeplabv3plus
```

Train only SegFormer-B2:

```bash
python semantic_models/train_semantic_models.py \
    --models segformerb2
```

Train a single model on a single fold:

```bash
python semantic_models/train_semantic_models.py \
    --models unetpp \
    --folds fold1
```

The default training configuration used in the paper is:

| Parameter     | Value |
| ------------- | ----: |
| Epochs        |    30 |
| Image size    |   512 |
| Batch size    |     8 |
| Learning rate |  1e-4 |
| Weight decay  |  1e-4 |

The models use the following pretrained encoders:

| Model        | Encoder   |
| ------------ | --------- |
| U-Net++      | ResNet-34 |
| DeepLabV3+   | ResNet-34 |
| SegFormer-B2 | MiT-B2    |

The corresponding ImageNet pretrained weights are downloaded automatically the first time each model is trained.

The principal options can be overridden, for example:

```bash
python semantic_models/train_semantic_models.py \
    --models segformerb2 \
    --folds fold1 \
    --dataset_root ~/ExoDisc_project/ExoDisc \
    --results_root ~/ExoDisc_results/semantic_benchmark
```

## Output

Training results are written to:

```text
~/ExoDisc_results/
└── semantic_benchmark/
```

Each model generates an independent directory:

```text
unetpp/
deeplabv3plus/
segformerb2/
```

For each fold, the following files are generated:

```text
best.pt
last.pt
classes.txt
config.json
results.csv
```

The training log (`results.csv`) reports the training and validation losses together with the corresponding mIoU values for every epoch. The best-performing model is selected according to the validation mIoU.
