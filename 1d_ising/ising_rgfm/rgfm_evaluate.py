"""Evaluate open-boundary two-point correlations for local RGFM on 1D Ising data."""

import argparse
import json
import os

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from model import LocalConvDenoiser1D
from rgfm_diffusion import RGDiffusionReal1D
from rgfm_sampler import RGSamplerReal, RGSamplerTotal


def get_chains(batch):
    if isinstance(batch, dict):
        if "x" in batch:
            return batch["x"]
        if "chain" in batch:
            return batch["chain"]
        raise KeyError("dict batch must contain key 'x' or 'chain'")
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def load_config(path):
    with open(path, "r") as f:
        return ConfigDict(json.load(f))


def build_model(flags, device):
    return LocalConvDenoiser1D(
        hidden_dim=flags.hidden_dim,
        time_dim=flags.time_dim,
        depth=flags.depth,
        kernel_size=flags.kernel_size,
    ).to(device)


def build_unified_sampler(model_files, model_key, device):
    entries = []
    for expected_length, config_path, checkpoint_path in model_files:
        flags = load_config(config_path)
        if int(flags.L) != expected_length:
            raise ValueError(
                f"Expected L={expected_length} in {config_path}, got L={flags.L}"
            )
        entries.append(
            {
                "config": config_path,
                "checkpoint": checkpoint_path,
                "flags": flags,
                "L": int(flags.L),
                "t_interval": tuple(flags.t_interval),
            }
        )

    entries.sort(key=lambda entry: entry["L"], reverse=True)
    reference = entries[0]["flags"]
    lengths = [entry["L"] for entry in entries]

    if lengths[0] != reference.base_L:
        raise ValueError("The largest model must have L equal to base_L.")

    shared_fields = (
        "base_L",
        "dataset_key",
        "data_path",
        "sigma_min",
        "mu_min",
        "mL",
        "reg_name",
    )
    for entry in entries:
        flags = entry["flags"]
        for field in shared_fields:
            if getattr(flags, field) != getattr(reference, field):
                raise ValueError(f"Inconsistent {field} in {entry['config']}")

    for current, following in zip(entries, entries[1:]):
        if current["L"] <= following["L"]:
            raise ValueError("RGFM resolutions must be strictly decreasing.")
        if abs(current["t_interval"][1] - following["t_interval"][0]) >= 1e-6:
            raise ValueError(
                f"Disconnected time intervals between L={current['L']} and L={following['L']}."
            )

    if abs(entries[0]["t_interval"][0]) >= 1e-6:
        raise ValueError("The largest-resolution interval must start at t=0.")
    if abs(entries[-1]["t_interval"][1] - 1.0) >= 1e-6:
        raise ValueError("The smallest-resolution interval must end at t=1.")

    diffusion = RGDiffusionReal1D(
        L=reference.base_L,
        sigma_min=reference.sigma_min,
        mu_min=reference.mu_min,
        mL=reference.mL,
        reg_name=reference.reg_name,
    ).to(device)

    samplers = []
    for entry in entries:
        flags = entry["flags"]
        model = build_model(flags, device)
        state = torch.load(entry["checkpoint"], map_location=device)
        model.load_state_dict(state[model_key])
        model.eval()

        samplers.append(
            RGSamplerReal(
                model=model,
                t_interval=entry["t_interval"],
                n_total_steps=flags.sample_n_steps,
                method=flags.sample_method,
                rtol=flags.sample_rtol,
                atol=flags.sample_atol,
            ).to(device)
        )
        print(
            f"Loaded L={entry['L']}, t_interval={entry['t_interval']}, "
            f"checkpoint={entry['checkpoint']}"
        )

    total_sampler = RGSamplerTotal(
        sampler_list=samplers,
        L_list=lengths,
        rg_diffusion_real=diffusion,
    ).to(device)
    total_sampler.eval()
    return total_sampler, reference, entries


@torch.no_grad()
def collect_data_samples(flags, num_samples, batch_size):
    dataset = get_dataset(
        flags.dataset_key,
        L=flags.base_L,
        data_path=getattr(flags, "data_path", None),
        max_samples=getattr(flags, "max_samples", None),
        normalize=getattr(flags, "normalize_data", False),
        subtract_mean=getattr(flags, "subtract_mean", False),
        mmap_mode=getattr(flags, "mmap_mode", "r"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=flags.num_workers,
        drop_last=False,
    )

    samples = []
    num_collected = 0
    for batch in tqdm(dataloader, desc="Collecting Ising data"):
        x = get_chains(batch)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1:] != (1, flags.base_L):
            raise ValueError(
                f"Expected samples with shape [B, 1, {flags.base_L}], got {tuple(x.shape)}"
            )

        keep = min(x.shape[0], num_samples - num_collected)
        if keep <= 0:
            break
        samples.append(x[:keep].cpu())
        num_collected += keep
        if num_collected == num_samples:
            break

    if not samples:
        raise RuntimeError("No data samples were collected.")
    return torch.cat(samples, dim=0)


@torch.no_grad()
def collect_generated_samples(sampler, num_samples, batch_size):
    samples = []
    num_collected = 0
    with tqdm(total=num_samples, desc="Generating RGFM samples") as progress:
        while num_collected < num_samples:
            current_batch = min(batch_size, num_samples - num_collected)
            samples.append(sampler(current_batch).cpu())
            num_collected += current_batch
            progress.update(current_batch)
    return torch.cat(samples, dim=0)


def two_point_correlation(x, max_distance=None):
    """Compute open-boundary C(r) = <x_i x_{i+r}> without binarization."""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected x with shape [N, 1, L], got {tuple(x.shape)}")

    length = x.shape[-1]
    if max_distance is None:
        max_distance = length - 1
    if not 0 <= max_distance < length:
        raise ValueError(f"max_distance must be between 0 and {length - 1}")

    correlations = []
    for r in range(max_distance + 1):
        if r == 0:
            correlations.append((x * x).mean())
        else:
            correlations.append((x[..., :-r] * x[..., r:]).mean())
    return torch.stack(correlations)


def save_correlation(c_data, c_model, out_dir):
    distances = np.arange(len(c_data))
    table = np.column_stack([distances, c_data.numpy(), c_model.numpy()])
    csv_path = os.path.join(out_dir, "correlation.csv")
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="r,C_data,C_model",
        comments="",
    )
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate open-boundary two-point correlations for local RGFM on 1D Ising."
    )
    parser.add_argument("--config_L1024", type=str, required=True)
    parser.add_argument("--ckpt_L1024", type=str, required=True)
    parser.add_argument("--config_L512", type=str, required=True)
    parser.add_argument("--ckpt_L512", type=str, required=True)
    parser.add_argument("--config_L256", type=str, required=True)
    parser.add_argument("--ckpt_L256", type=str, required=True)
    parser.add_argument("--config_L128", type=str, required=True)
    parser.add_argument("--ckpt_L128", type=str, required=True)
    parser.add_argument("--config_L64", type=str, required=True)
    parser.add_argument("--ckpt_L64", type=str, required=True)
    parser.add_argument("--config_L32", type=str, required=True)
    parser.add_argument("--ckpt_L32", type=str, required=True)
    parser.add_argument("--config_L16", type=str, required=True)
    parser.add_argument("--ckpt_L16", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--max_distance", type=int, default=None)
    parser.add_argument(
        "--model_key",
        type=str,
        default="ema_model",
        choices=["ema_model", "net_model"],
    )
    args = parser.parse_args()

    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive.")

    model_files = [
        (1024, args.config_L1024, args.ckpt_L1024),
        (512, args.config_L512, args.ckpt_L512),
        (256, args.config_L256, args.ckpt_L256),
        (128, args.config_L128, args.ckpt_L128),
        (64, args.config_L64, args.ckpt_L64),
        (32, args.config_L32, args.ckpt_L32),
        (16, args.config_L16, args.ckpt_L16),
    ]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    sampler, reference, entries = build_unified_sampler(model_files, args.model_key, device)
    out_dir = args.out_dir or os.path.join(reference.log_root, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    x_data = collect_data_samples(reference, args.num_samples, args.batch_size)
    x_model = collect_generated_samples(sampler, args.num_samples, args.batch_size)
    c_data = two_point_correlation(x_data, args.max_distance)
    c_model = two_point_correlation(x_model, args.max_distance)
    csv_path = save_correlation(c_data, c_model, out_dir)

    summary = {
        "num_samples": int(args.num_samples),
        "base_L": int(reference.base_L),
        "max_distance": int(len(c_data) - 1),
        "model_key": args.model_key,
        "resolutions": [entry["L"] for entry in entries],
        "t_intervals": [list(entry["t_interval"]) for entry in entries],
        "configs": [entry["config"] for entry in entries],
        "checkpoints": [entry["checkpoint"] for entry in entries],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Correlation evaluation finished.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
