# 1D Ising Data Generation

This directory contains local renormalization-group flow matching (local RGFM) and local flow matching (local FM) experiments for the one-dimensional Ising model. Configs are included for correlation lengths `xi = 32, 64, 128, 256`. The commands below use `L = 1024` and `xi = 128` as an example.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm
- ml-collections
- torchdiffeq
- torch-dct

## Dataset

The Ising dataset must be stored as a NumPy `.npy` array with shape `(N, L)` or `(N, 1, L)`. The expected samples take values in `{-1, +1}`. Update `data_path` in the configs if the dataset is stored elsewhere.

The script `other_utils/1Dising_dataset_make.py` can be used to generate these datasets. For the numerical experiments reported in the paper, we generated 100,000 samples for each correlation length:

```bash
python 1d_ising/other_utils/1Dising_dataset_make.py \
  --xi_list 32 64 128 256 \
  --out_dir 1d_ising/dataset_ising1d_open
```

## Checkpoints

The checkpoints used in the paper is provided [here](url). To run the code with the excecution below, the checkpoint directory should be placed as 

```
REPOSITORY_DIRECTORY/check_points/ising/...
```

## Execution Scripts

Execution scripts are stored in `exe_files/`. Run the commands below from the repository root. The optional final argument is the GPU index and defaults to `0`.

## Local RGFM: base L1024, xi=128

The RGFM setup uses seven models at `L = 1024, 512, 256, 128, 64, 32, 16`.

### Training

Train all seven models sequentially:

```bash
bash 1d_ising/exe_files/rgfm_train_baseL1024_xi128.sh 0
```

### Sampling

```bash
bash 1d_ising/exe_files/rgfm_sample_baseL1024_xi128.sh 0
```

The sampling and evaluation scripts expect `L1024.pt`, `L512.pt`, `L256.pt`, `L128.pt`, `L64.pt`, `L32.pt`, and `L16.pt` under:

```text
check_points/ising/rgfm_baseL1024_xi128/
```

### Evaluation

```bash
bash 1d_ising/exe_files/rgfm_evaluate_baseL1024_xi128.sh 0
```

## Local FM: L1024, xi=128

### Training

```bash
bash 1d_ising/exe_files/fm_train_L1024_xi128.sh 0
```

### Sampling

```bash
bash 1d_ising/exe_files/fm_sample_L1024_xi128.sh 0
```

The sampling and evaluation scripts expect:

```text
check_points/ising/fm_L1024_xi128.pt
```

### Evaluation

```bash
bash 1d_ising/exe_files/fm_evaluate_L1024_xi128.sh 0
```

## Outputs

Training outputs are written under the `logdir` specified in each config, with checkpoints in its `ckpt/` subdirectory. Sampling saves generated tensors. Evaluation computes the open-boundary real-space two-point correlation function directly from the generated values and saves a CSV file and a JSON summary.
