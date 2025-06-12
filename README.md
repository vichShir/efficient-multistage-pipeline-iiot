# An Efficient Feature Selection Pipeline for IIoT Threat Detection

This repository contains all source code used for the work of "An Efficient Feature Selection Pipeline for IIoT Threat Detection".

For more information contact: victor.shirasuna@ime.usp.br

## Table of Contents

1. [Getting Started](#getting-started)
	1. [Directory Organization](#directory-organization)
    2. [Replicating Python Environment - Pipeline](#replicating-python-environment---pipeline)
    3. [Replicating Python Environment - Raspberry Pi](#replicating-python-environment---raspberry-pi)

## Getting Started

Follow these steps to replicate our Mamba environment and install the necessary libraries:

### Directory Organization

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