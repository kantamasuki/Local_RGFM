# Image Generation with Local RGFM

This directory contains local renormalization-group flow matching (local RGFM) and local flow matching (local FM) experiments for image generation. Configs are included for `L = 64` with patch sizes `P = 32, 48, 64`, and for `L = 256` with `P = 32`.

## Requirements

- PyTorch
- torchvision
- NumPy
- SciPy
- tqdm
- ml-collections
- torchdiffeq
- torch-dct
- clean-fid

## Dataset

The experiments use the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) at resolutions 64 and 256. The dataset preparation script is provided at `datasets/resize_ffhq.py`. Place the downloaded source directories under `2d_image/datasets/`; the script selects the appropriate source directory from `--img_size` automatically.

### FFHQ64

Download the FFHQ `thumbnails128x128` directory and place it under `2d_image/datasets/`. The following command resizes all of its PNG images to 64 x 64:

```bash
cd 2d_image/datasets
python resize_ffhq.py --img_size 64
```

The resized images are saved under `datasets/ffhq64/`.

### FFHQ256

Download the FFHQ `images1024x1024` directory and place it under `2d_image/datasets/`. The following command selects the first 5,000 PNG files in filename order and resizes them from 1024 x 1024 to 256 x 256:

```bash
cd 2d_image/datasets
python resize_ffhq.py --img_size 256
```

The resized images are saved under `datasets/ffhq256/`. Only these first 5,000 images were used for the 256 x 256 numerical experiments.

## Preparation for FID Evaluation

FID evaluation requires clean-FID statistics for the original real images. These statistics can be created with `other_utils/fid_stats_maker.py`:

```bash
python 2d_image/other_utils/fid_stats_maker.py \
  --data_dir 2d_image/datasets/ffhq64 \
  --num_images 50000 \
  --out_path 2d_image/FID_features/ffhq64_raw_L64_cleanfid.npz \
  --device 0
```

This utility extracts clean-FID Inception features directly from the original dataset images and saves their mean and covariance. It does not apply resolution changes.

The evaluation scripts expect the resulting statistics at:

```text
2d_image/FID_features/ffhq64_raw_L64_cleanfid.npz
```

## Execution Scripts

Execution scripts are stored in `exe_files/`. Run the commands below from the repository root. The optional final argument is the GPU index and defaults to `0`.

## Local RGFM

For the `L = 64, P = 48` and `L = 64, P = 64` setups, only the
largest-resolution model is patch-size specific. Their complete RGFM chains
reuse the `L = 32` and `L = 16` configs and checkpoints from the `P = 32`
setup. To run either variant, replace the `L = 64` config and checkpoint in
the P32 commands while keeping the P32 `L = 32` and `L = 16` entries.

### Training: base L64, P32

Train the L64, L32, and L16 models sequentially:

```bash
bash 2d_image/exe_files/rgfm_train_baseL64_P32.sh 0
```

### Sampling: base L64, P32

```bash
bash 2d_image/exe_files/rgfm_sample_baseL64_P32.sh 0
```

The sampling and evaluation scripts expect `L64.pt`, `L32.pt`, and `L16.pt` under:

```text
check_points/image/rgfm_baseL64_P32/
```

### Evaluation: base L64, P32

```bash
bash 2d_image/exe_files/rgfm_evaluate_baseL64_P32.sh 0
```

### Sampling: base L256, P32

L256 sampling uses the separate five-stage program `image_rgfm/rgfm_sample_L256.py`:

```bash
bash 2d_image/exe_files/rgfm_sample_baseL256_P32.sh 0
```

This script expects `L256.pt`, `L128.pt`, `L64.pt`, `L32.pt`, and `L16.pt` under:

```text
check_points/image/rgfm_baseL256_P32/
```

No base L256 evaluation script is included.

## Local FM

The FM configs `fm_L64_P48.json`, `fm_L64_P64.json`, and
`fm_L256_P32.json` can be passed to the same training and sampling programs
as the P32 example below, together with their corresponding checkpoints.

### Training: L64, P32

```bash
bash 2d_image/exe_files/fm_train_L64_P32.sh 0
```

### Sampling: L64, P32

```bash
bash 2d_image/exe_files/fm_sample_L64_P32.sh 0
```

The sampling and evaluation scripts expect:

```text
check_points/image/fm_L64_P32.pt
```

### Evaluation: L64, P32

```bash
bash 2d_image/exe_files/fm_evaluate_L64_P32.sh 0
```

## Outputs

Training outputs are written under the `logdir` specified in each config, with checkpoints in its `ckpt/` subdirectory. Sampling saves generated image grids. Evaluation saves clean-FID generated statistics, preview images, and a CSV result.
