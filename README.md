# An Efficient Multi-Stage Pipeline for IIoT Threat Detection

This repository contains all source code used for the work of "An Efficient Multi-Stage Pipeline for IIoT Threat Detection".

For more information contact: victor.shirasuna@ime.usp.br

![Multi-Stage Pipeline](./images/multi-stage_pipeline.png)

## Table of Contents

1. [Getting Started](#1-getting-started)
    1. [Directory Organization](#directory-organization)
    2. [Replicating Python Environment](#replicating-python-environment)
    4. [Configuring the BRURIIoT dataset](#configuring-the-bruriiot-dataset)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Running the Pipeline](#3-running-the-pipeline)
4. [Running Predictions on Raspberry Pi](#4-running-predictions-on-raspberry-pi)

## 1. Getting Started

Follow these steps to replicate our data organization and Python environment:

### Directory Organization

#### Multi-Stage Pipeline Structure

Execute the following script to create the data directories:
```shell
cd ./scripts
bash run_create_dirs.sh
```

The following directories will be created:
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

### Replicating Python Environment

#### Install Miniforge

Install miniforge with conda version `24.9.2` and mamba version `1.5.9` from: https://github.com/conda-forge/miniforge/releases/tag/24.9.2-0

#### Create and Activate Mamba Environment

```shell
mamba create -n pipeline python=3.10.16
mamba activate pipeline
```

#### Install Pip dependencies

For the `multi-stage pipeline` environment, use the following command:
```shell
pip install -r requirements_pipeline.txt
```

For the `Raspberry Pi` environment, use the following command:
```shell
pip install -r requirements_raspberrypi.txt
```

### Configuring the BRURIIoT dataset

The `BRURIIoT` dataset can be downloaded at the following link: https://ieee-dataport.org/documents/bruriiot-dataset-network-anomaly-detection-iiot-enhanced-feature-engineering-approach

Place the `BRUIIoT.csv` dataset file at the following location:
```
data/
└── datasets/
    ├── splits/
    ├── features/
    └── BRUIIoT.csv
```

> The pipeline will automatically create the necessary data splits and place at `data/datasets/splits` path.

## 2. Exploratory Data Analysis (EDA)

We conducted an Exploratory Data Analysis (EDA) of the BRURIIoT dataset to further understand its structure. All used codes are available at the following notebook `exploratory_data_analysis_eda.ipynb`.

## 3. Running the Pipeline

## 4. Running Predictions on Raspberry Pi