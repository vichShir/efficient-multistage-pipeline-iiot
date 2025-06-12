# An Efficient Feature Selection Pipeline for IIoT Threat Detection

This repository contains all source code used for the work of "An Efficient Feature Selection Pipeline for IIoT Threat Detection".

For more information contact: victor.shirasuna@ime.usp.br

## Table of Contents

1. [Getting Started](#getting-started)
	1. [Directory Organization](#directory-organization)
    2. [Replicating Python Environment - Pipeline](#replicating-python-environment---pipeline)
    3. [Replicating Python Environment - Raspberry Pi](#replicating-python-environment---raspberry-pi)
    4. [Configuring BRURIIoT dataset](#configuring-bruriiot-dataset)
2. [Running the Pipeline](#running-the-pipeline)
3. [Running Predictions on Raspberry Pi](#running-predictions-on-raspberry-pi)

## Getting Started

Follow these steps to replicate our data organization and Python environment with the necessary libraries:

### Directory Organization

#### Create Folders for Pipeline

The pipeline automatically creates the following folders:
```
data/
├── datasets/
│   ├── splits
│   └── features
├── models/
│   ├── training_size
│   └── feature_selection
└── results/
    ├── confusion_matrix
    ├── feature_importance
    ├── feature_selection
    ├── inference_time
    ├── training_size
    └── feature_selection/
        ├── features
        ├── metrics
        └── pipeline
```

#### Create Folders for Raspberry Pi

Create the following folders to run on a Raspberry Pi device:
```
raspberry_pi/
├── datasets
├── models
└── results
```

### Replicating Python Environment - Pipeline

#### Install Miniforge

Install miniforge with conda version `24.9.2` and mamba version `1.5.9` from: https://github.com/conda-forge/miniforge/releases/tag/24.9.2-0

#### Create and Activate Mamba Environment

```shell
mamba create -n pipeline python=3.10.16
mamba activate pipeline
```

#### Install Pip dependencies

```shell
pip install -r requirements_pipeline.txt
```

### Replicating Python Environment - Raspberry Pi

#### Install Miniforge

Install miniforge with conda version `24.9.2` and mamba version `1.5.9` from: https://github.com/conda-forge/miniforge/releases/tag/24.9.2-0

#### Create and Activate Mamba Environment

```shell
mamba create -n raspberry python=3.10.16
mamba activate raspberry
```

#### Install Pip dependencies

For Raspberry Pi environment, only the following python dependencies are required from `requirements.txt`:
```shell
pip install -r requirements_raspberry.txt
```

### Configuring BRURIIoT dataset

The BRURIIoT dataset can be downloaded at the following link: https://ieee-dataport.org/documents/bruriiot-dataset-network-anomaly-detection-iiot-enhanced-feature-engineering-approach

Place the `BRUIIoT.csv` dataset file at the following location:
```
data/
└── datasets/
    ├── splits
    ├── features
    └── BRUIIoT.csv
```

The pipeline will automatically create the necessary data splits and place at `data/datasets/splits` path.

### Running the Pipeline

### Running Predictions on Raspberry Pi