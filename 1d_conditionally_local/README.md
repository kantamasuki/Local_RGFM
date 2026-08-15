# 1D Conditionally Local Data Generation

This directory contains local renormalization-group flow matching (local RGFM) and local flow matching (local FM) experiments for a one-dimensional conditionally local toy distribution.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm
- ml-collections
- torchdiffeq
- torch-dct


## Dataset

The conditionally local dataset is generated on the fly by `dataset.py`. No external dataset file is required.

## Checkpoints

The checkpoints used in the paper is provided [here](url). To run the code with the excecution below, the checkpoint directory should be placed as 

```
REPOSITORY_DIRECTORY/check_points/cond_loc/...
```

## Execution Scripts

Execution scripts are stored in `exe_files/`. Run the commands below from the repository root. The optional final argument is the GPU index and defaults to `0`.

## Local RGFM

The RGFM setup uses seven models at `L = 1024, 512, 256, 128, 64, 32, 16`.

### Training

Train all seven models sequentially:

```bash
bash 1d_conditionally_local/exe_files/rgfm_train.sh 0
```

### Sampling

```bash
bash 1d_conditionally_local/exe_files/rgfm_sample.sh 0
```

The sampling and evaluation scripts expect `L1024.pt`, `L512.pt`, `L256.pt`, `L128.pt`, `L64.pt`, `L32.pt`, and `L16.pt` under:

```text
check_points/cond_loc/rgfm_baseL1024/
```

### Evaluation

```bash
bash 1d_conditionally_local/exe_files/rgfm_evaluate.sh 0
```

## Local FM

### Training

```bash
bash 1d_conditionally_local/exe_files/fm_train.sh 0
```

### Sampling

```bash
bash 1d_conditionally_local/exe_files/fm_sample.sh 0
```

The sampling and evaluation scripts expect:

```text
check_points/cond_loc/fm_L1024.pt
```

### Evaluation

```bash
bash 1d_conditionally_local/exe_files/fm_evaluate.sh 0
```

## Outputs

Outputs are written under the `logdir` specified in each config. Training saves checkpoints in its `ckpt/` subdirectory. Sampling saves generated tensors. Evaluation computes the real-space two-point correlation function and saves a CSV file.
