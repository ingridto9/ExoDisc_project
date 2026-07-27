# ExoDisc Project

Code accompanying the ExoDisc dataset descriptor:

> **ExoDisc: An annotated dataset for surgical instrument and anatomy segmentation in open lumbar microdiscectomy**
> Ingrid Tombini, Domenico Pachioli, Mattia Magro, Hans Schoepp, Antoine Pfeil, Janina Fritzenschaft, Niccolò Innocenti, Francesco Costa, Elena De Momi
> *Scientific Data* (submitted) — DOI: *to be added*

This repository contains the code used to reproduce the benchmark experiments presented in the paper, including model training, inference, and evaluation. The ExoDisc dataset (RGB images, COCO-format instance annotations, and semantic segmentation masks) is distributed separately through Zenodo.

## Repository structure

| Folder                                 | Model(s)                          | Description                                                                                 |
| -------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| [`semantic_models/`](semantic_models/) | U-Net++, DeepLabV3+, SegFormer-B2 | Training and evaluation of three fully supervised semantic segmentation models              |
| [`mask2former/`](mask2former/)         | Mask2Former                       | Training and evaluation of the Hugging Face implementation                                  |
| [`dinov2/`](dinov2/)                   | DINOv2                            | Training and evaluation of a DINOv2-based semantic segmentation model                       |
| [`yolo11/`](yolo11/)                   | YOLO11m-seg                       | Instance segmentation training and semantic benchmark evaluation                            |
| [`medsam/`](medsam/)                   | MedSAM                            | Prompt-guided evaluation using ground-truth bounding-box and point prompts (no fine-tuning) |

Each directory can be used independently. It contains its own training/evaluation script(s), a dedicated `README.md` describing the setup and usage, and a lock file documenting the software environment used to generate the benchmark results reported in the paper.

## Implemented benchmarks

| Model        | Training | Evaluation |
| ------------ | :------: | :--------: |
| U-Net++      |     ✓    |      ✓     |
| DeepLabV3+   |     ✓    |      ✓     |
| SegFormer-B2 |     ✓    |      ✓     |
| DINOv2       |     ✓    |      ✓     |
| Mask2Former  |     ✓    |      ✓     |
| YOLO11m-seg  |     ✓    |      ✓     |
| MedSAM       |     –    |      ✓     |

## Setup

### 1. Create the project directory

```bash
mkdir ~/ExoDisc_project
```

### 2. Download the ExoDisc dataset

Download the ExoDisc dataset from Zenodo and extract it inside `~/ExoDisc_project`.

The default directory structure is:

```text
~/ExoDisc_project/
└── ExoDisc/
    ├── op1/
    ├── op2/
    ├── op3/
    └── op4/
```

Each operation folder contains the RGB images, COCO-format instance annotations, and the corresponding semantic segmentation masks.

### 3. Choose the benchmark

Each benchmark is completely independent.

Select the model you want to reproduce and follow the instructions in the corresponding directory:

* `semantic_models/`
* `mask2former/`
* `dinov2/`
* `yolo11/`
* `medsam/`

Each directory provides:

* software installation instructions;
* required pretrained models (when applicable);
* training and/or evaluation commands;
* expected output files.

### 4. Install the required software

The benchmark pipelines were developed using different deep learning frameworks and software environments. Therefore, **no global environment is provided**.

Each model directory contains:

* a dedicated `README.md` with installation instructions;
* a `requirements-*-lock.txt` file documenting the software environment used for the experiments reported in the paper.

Install **only** the environment required for the model(s) you intend to reproduce.

### 5. Run the benchmark

Follow the commands reported in the corresponding model directory to reproduce the experiments.

## Evaluation protocol

All fully automatic models (U-Net++, DeepLabV3+, SegFormer-B2, DINOv2, Mask2Former, and YOLO11m-seg) were evaluated using **patient-wise four-fold leave-one-operation-out cross-validation**, where one complete surgical procedure is held out for testing in each fold.

MedSAM is evaluated separately as a prompt-guided reference method without fine-tuning, using ground-truth bounding-box and point prompts derived from the released annotations.

## Notes

The semantic segmentation masks required by the semantic segmentation benchmarks are already distributed with the ExoDisc dataset.

No additional preprocessing is required before running the provided training and evaluation pipelines.

## Citation

If you use the ExoDisc dataset or the accompanying benchmark code in your research, please cite:

```text
Citation to be added after publication.
```

## License

- **Code:** MIT License.
- **Dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

**Ingrid Tombini**

Department of Electronics, Information and Bioengineering (DEIB)
Politecnico di Milano, Italy

📧 [ingrid.tombini@polimi.it](mailto:ingrid.tombini@polimi.it)
