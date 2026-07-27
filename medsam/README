# MedSAM

This directory contains the evaluation pipeline used to reproduce the MedSAM benchmarking experiments reported in the ExoDisc paper.

The benchmark evaluates MedSAM using ground-truth prompts under two prompting strategies:

- Bounding box prompts
- Point prompts

No training is required.

---

## Dataset

Download the ExoDisc dataset from Zenodo and extract it as

```text
~/ExoDisc_project/
└── ExoDisc/
    ├── op1/
    ├── op2/
    ├── op3/
    └── op4/
```

---

## MedSAM checkpoint

The pretrained MedSAM checkpoint is **not distributed** with this repository.

Please download the official MedSAM ViT-B checkpoint from the official MedSAM repository:

https://github.com/bowang-lab/MedSAM

and place it in

```text
~/ExoDisc_project/
└── checkpoints/
    └── medsam/
        └── medsam_vit_b.pth
```

Alternatively, specify a different location using

```bash
--checkpoint /path/to/medsam_vit_b.pth
```

---

## Software environment

The benchmark was originally developed using the software environment reported in

```text
requirements-mask2former-medsam-dinov2-lock.txt
```

The file documents the exact package versions and MedSAM commit used to reproduce the experiments reported in the paper.

---

## Evaluation

Run all folds using both prompting strategies:

```bash
python evaluate_medsam_prompting.py
```

Run only bounding-box prompting:

```bash
python evaluate_medsam_prompting.py \
    --prompt_modes bbox
```

Run only point prompting:

```bash
python evaluate_medsam_prompting.py \
    --prompt_modes point
```

Evaluate a single fold:

```bash
python evaluate_medsam_prompting.py \
    --fold fold1
```

Save the predicted semantic masks:

```bash
python evaluate_medsam_prompting.py \
    --save_masks
```

---

## Output

Results are written to

```text
~/ExoDisc_results/
└── semantic_benchmark/
    └── medsam_prompting/
```

including

- fold-level metrics
- per-class metrics
- summary statistics
- predicted semantic masks (optional)
