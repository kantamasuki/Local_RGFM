"""Evaluate one local FM checkpoint with clean-FID."""

import argparse
import csv
import json
import os
import random
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.resize import build_resizer
from scipy import linalg
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from fm_sampler import FMSamplerReal
from model import UNet
from patch_maker import PatchMaker


def load_config(path):
    with open(path, "r") as f:
        return SimpleNamespace(**json.load(f))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if not torch.cuda.is_available() or device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        return torch.device(device_arg)
    return torch.device(f"cuda:{device_arg}")


def build_model(config, device):
    return UNet(
        ch=config.ch,
        ch_mult=config.ch_mult,
        attn=config.attn,
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout,
        in_channels=5,
        out_channels=3,
    ).to(device)


def load_model_weight(model, checkpoint_path, device, key="ema_model"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    if key in checkpoint:
        state = checkpoint[key]
    elif "ema_model" in checkpoint:
        state = checkpoint["ema_model"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "net_model" in checkpoint:
        state = checkpoint["net_model"]
    else:
        state = checkpoint

    state = {
        name.removeprefix("module."): value
        for name, value in state.items()
    }
    model.load_state_dict(state)


def build_sampler(config, checkpoint_path, device, checkpoint_key):
    model = build_model(config, device)
    load_model_weight(model, checkpoint_path, device, key=checkpoint_key)
    model.eval()

    patch_maker = PatchMaker(
        L=config.img_size,
        P=config.P,
        D=config.D,
        stride=config.stride,
        normalize_coord=config.normalize_coord,
    )
    sampler = FMSamplerReal(
        model=model,
        patch_maker=patch_maker,
        img_size=config.img_size,
        n_steps=config.sample_n_steps,
        method=config.sample_method,
        rtol=config.sample_rtol,
        atol=config.sample_atol,
    ).to(device)
    sampler.eval()
    return sampler


def to_clean_fid_input(images, resize):
    images = (images.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    images = torch.clamp(images * 255.0, 0, 255).to(torch.uint8)

    resized = torch.empty(images.shape[0], 3, 299, 299, dtype=torch.float32)
    for index, image in enumerate(images):
        image_array = image.numpy().transpose(1, 2, 0)
        resized_array = resize(image_array)
        resized[index] = torch.from_numpy(
            resized_array.transpose(2, 0, 1)
        ).float()
    return resized


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariances have different shapes: {sigma1.shape} vs {sigma2.shape}")

    difference = mu1 - mu2
    covariance_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covariance_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real

    return float(
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covariance_mean)
    )


def load_real_stats(path):
    stats = np.load(path)
    if "mu" not in stats or "sigma" not in stats:
        raise ValueError(f"{path} must contain 'mu' and 'sigma'")
    return stats["mu"], stats["sigma"]


def save_preview(images, path, nrow=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if nrow is None:
        nrow = max(1, int(images.shape[0] ** 0.5))
    grid = make_grid(
        (images.detach().cpu().clamp(-1, 1) + 1.0) / 2.0,
        nrow=nrow,
    )
    save_image(grid, path)


@torch.no_grad()
def collect_generated_features(
    sampler,
    feature_model,
    resize,
    sample_device,
    feature_device,
    num_images,
    batch_size,
    save_batches,
    preview_dir,
    nrow,
):
    features = []
    num_generated = 0
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_index in tqdm(range(num_batches), desc="generate + feature"):
        current_batch = min(batch_size, num_images - num_generated)
        if current_batch <= 0:
            break

        images = sampler(
            batch_size=current_batch,
            channels=3,
            device=sample_device,
        )
        if batch_index < save_batches:
            save_preview(
                images,
                os.path.join(preview_dir, f"batch{batch_index:04d}_generated.png"),
                nrow=nrow,
            )

        fid_input = to_clean_fid_input(images, resize).to(
            feature_device, non_blocking=True
        )
        features.append(feature_model(fid_input).detach().cpu().numpy())
        num_generated += current_batch

    if not features:
        raise RuntimeError("No generated features were collected")
    return np.concatenate(features, axis=0)


def save_generated_stats(path, mu, sigma, features, metadata, save_features):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "mu": mu,
        "sigma": sigma,
        "metadata_json": json.dumps(metadata, indent=2, sort_keys=True),
    }
    if save_features:
        payload["features"] = features
    np.savez(path, **payload)


def save_result(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one local FM checkpoint with clean-FID."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ckpt_key", type=str, default="ema_model")
    parser.add_argument("--fid_stats", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--fid_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="fm_fid_eval")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--save_batches", type=int, default=1)
    parser.add_argument("--nrow", type=int, default=None)
    args = parser.parse_args()

    if args.num_images <= 1:
        raise ValueError("num_images must be greater than 1 to compute covariance")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(args.seed)
    sample_device = resolve_device(args.device)
    feature_device = (
        resolve_device(args.fid_device)
        if args.fid_device is not None
        else sample_device
    )

    config = load_config(args.config)
    sampler = build_sampler(
        config, args.ckpt, sample_device, args.ckpt_key
    )
    real_mu, real_sigma = load_real_stats(args.fid_stats)
    feature_model = build_feature_extractor(mode="clean", device=feature_device)
    feature_model.eval()
    resize = build_resizer(mode="clean")

    os.makedirs(args.out_dir, exist_ok=True)
    features = collect_generated_features(
        sampler=sampler,
        feature_model=feature_model,
        resize=resize,
        sample_device=sample_device,
        feature_device=feature_device,
        num_images=args.num_images,
        batch_size=args.batch_size,
        save_batches=args.save_batches,
        preview_dir=os.path.join(args.out_dir, "previews"),
        nrow=args.nrow,
    )
    generated_mu = np.mean(features, axis=0)
    generated_sigma = np.cov(features, rowvar=False)
    fid = frechet_distance(generated_mu, generated_sigma, real_mu, real_sigma)

    metadata = {
        "config": args.config,
        "checkpoint": args.ckpt,
        "checkpoint_key": args.ckpt_key,
        "fid_stats": args.fid_stats,
        "num_images": int(features.shape[0]),
        "seed": args.seed,
    }
    stats_path = os.path.join(args.out_dir, "generated_stats.npz")
    save_generated_stats(
        stats_path,
        generated_mu,
        generated_sigma,
        features,
        metadata,
        args.save_features,
    )

    csv_path = args.out_csv or os.path.join(args.out_dir, "results.csv")
    row = {
        "img_size": int(config.img_size),
        "P": int(config.P),
        "fid": fid,
        "num_images": int(features.shape[0]),
        "fid_stats": args.fid_stats,
        "generated_stats": stats_path,
        "config": args.config,
        "checkpoint": args.ckpt,
        "seed": args.seed,
    }
    save_result(csv_path, row)

    print(f"FID: {fid:.6f}")
    print(f"Generated statistics: {stats_path}")
    print(f"Result CSV: {csv_path}")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()
