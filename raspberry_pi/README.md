# Experiment with Raspberry Pi 4

## Python Environment Reproduction

### Miniforge
Install miniforge with conda version `24.9.2` and mamba version `1.5.9` from: https://github.com/conda-forge/miniforge/releases/tag/24.9.2-0

Create conda environment:
```shell
mamba create -n iot-rasp python=3.10.16
mamba activate iot-rasp
```

### Python dependencies
For Raspberry Pi environment, only the following python dependencies are required from `requirements.txt`:
```shell
numpy>=2.2.5
pandas>=2.2.3
scikit-learn>=1.6.1
xgboost==2.1.4
tqdm>=4.67.1
```

## Directory organization

Create the following folders:
```
datasets
models
results
```