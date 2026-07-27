# YOLO11m-seg benchmark

This directory contains the code used to train and evaluate YOLO11m-seg on the ExoDisc patient-wise four-fold benchmark.

## Expected directory structure

The repository and the ExoDisc dataset must be arranged as follows by default:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   │   ├── images/
│   │   └── labels/
│   ├── op2/
│   ├── op3/
│   └── op4/
└── yolo11/
    ├── train_yolo11m_fold.py
    ├── evaluate_yolo_semantic.py
    └── requirements-yolo11-lock.txt
```

The scripts also accept custom project and results paths through command-line arguments.

## Software environment

`requirements-yolo11-lock.txt` records the exact principal package versions resolved in the Singularity container used for the experiments reported in the paper.

The original environment used:

- PyTorch 2.11.0 with CUDA 12.8;
- torchvision 0.26.0 with CUDA 12.8;
- torchaudio 2.11.0 with CUDA 12.8;
- Ultralytics 8.4.90 at commit `f9a0a334a9427366251b0b8bf93c569e887f21c7`.

A reproducible installation can be created with:

```bash
python -m venv .venv-yolo11
source .venv-yolo11/bin/activate
python -m pip install --upgrade pip

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install numpy==2.4.3 \
  opencv-python-headless==5.0.0.93 \
  pyyaml==6.0.3 \
  scipy==1.18.0

git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout f9a0a334a9427366251b0b8bf93c569e887f21c7
pip install -e .
cd ..
```

A CUDA-capable NVIDIA GPU is recommended. The exact CUDA build above reproduces the original software environment; users running a different CUDA version must install the corresponding PyTorch build.

## Cross-validation folds

The benchmark uses patient-wise leave-one-operation-out cross-validation:

| Fold | Training operations | Validation operation |
|---|---|---|
| fold1 | op2, op3, op4 | op1 |
| fold2 | op1, op3, op4 | op2 |
| fold3 | op1, op2, op4 | op3 |
| fold4 | op1, op2, op3 | op4 |

The benchmark contains 12 foreground classes. The `lamina` class is excluded, and the original YOLO class identifiers are remapped automatically by the training script.

## Training

Run one fold from the repository root:

```bash
python yolo11/train_yolo11m_fold.py --fold fold1
```

Repeat the command for `fold2`, `fold3`, and `fold4`.

The default training configuration used in the paper is:

| Parameter | Value |
|---|---:|
| Model | `yolo11m-seg.pt` |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 16 |
| Patience | 30 |

The script first creates the YOLO-formatted fold dataset under:

```text
~/ExoDisc_project/experiments/yolo/<fold>/
```

Images are linked rather than copied. Training outputs are saved under:

```text
~/ExoDisc_results/yolo11/<fold>/
```

The principal options can be overridden, for example:

```bash
python yolo11/train_yolo11m_fold.py \
  --fold fold1 \
  --dataset-root ~/ExoDisc_project/ExoDisc \
  --project-root ~/ExoDisc_project \
  --results-root ~/ExoDisc_results
```

## Semantic evaluation

After all folds have been trained, evaluate their `best.pt` checkpoints with:

```bash
python yolo11/evaluate_yolo_semantic.py --fold all
```

To evaluate a single fold:

```bash
python yolo11/evaluate_yolo_semantic.py --fold fold1
```

The default inference settings are:

| Parameter | Value |
|---|---:|
| Image size | 640 |
| Confidence threshold | 0.25 |
| NMS IoU threshold | 0.70 |

The evaluator merges YOLO instance predictions into semantic masks and reports foreground mIoU, mean Dice, mean class accuracy, foreground frequency-weighted IoU, background IoU, and per-class metrics.

Evaluation outputs are written to:

```text
~/ExoDisc_results/semantic_benchmark/yolo11m_seg/
```

For each fold, the script generates:

```text
<fold>_metrics.csv
<fold>_per_class.csv
<fold>_confusion_matrix.csv
```

When `--fold all` is used, it also generates:

```text
summary_folds.csv
summary_mean_std.csv
```
