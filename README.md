# ExoDisc Project

This repository contains the code used to reproduce the benchmark experiments presented in the ExoDisc dataset paper, including model training, inference, and evaluation.

## Setup

### 1. Create the project directory

Create a directory named `ExoDisc_project` in your home directory:

```bash
mkdir ~/ExoDisc_project
```

### 2. Download the ExoDisc dataset

Download the ExoDisc dataset from Zenodo and extract its contents inside `~/ExoDisc_project`.

The resulting directory structure should be:

```text
~/ExoDisc_project/
└── ExoDisc/
    ├── op1/
    ├── op2/
    ├── op3/
    ├── op4/
    └── ...
```

where each operation folder contains the RGB images, COCO annotations, and the provided semantic segmentation masks.

### 3. Clone this repository

```bash
cd ~/ExoDisc_project
git clone https://github.com/ingridto9/ExoDisc_project.git
```

The final directory structure should be:

```text
~/ExoDisc_project/
├── ExoDisc/
│   ├── op1/
│   ├── op2/
│   ├── op3/
│   └── ...
└── ExoDisc_project/
```
