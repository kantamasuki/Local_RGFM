"""Evaluate real-space two-point correlations for local FM."""

import argparse
import json
import os

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from fm_sampler import FMSamplerReal
from model import LocalConvDenoiser1D


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


def build_sampler(flags, checkpoint, model_key, device):
    model = LocalConvDenoiser1D(
        hidden_dim=flags.hidden_dim,
        time_dim=flags.time_dim,
        depth=flags.depth,
        kernel_size=flags.kernel_size,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state[model_key])
    model.eval()

    return FMSamplerReal(
        model=model,
        n_total_steps=flags.sample_n_steps,
        method=flags.sample_method,
        rtol=flags.sample_rtol,
        atol=flags.sample_atol,
    ).to(device)


@torch.no_grad()
def collect_data_samples(flags, num_samples, batch_size):
    dataset = get_dataset(flags.dataset_key, L=flags.L)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=flags.num_workers,
        drop_last=False,
    )

    samples = []
    num_collected = 0
    for batch in tqdm(dataloader, desc="Collecting data samples"):
        x = get_chains(batch)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1:] != (1, flags.L):
            raise ValueError(f"Expected samples with shape [B, 1, {flags.L}], got {tuple(x.shape)}")

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
def collect_generated_samples(sampler, length, device, num_samples, batch_size):
    samples = []
    num_collected = 0
    with tqdm(total=num_samples, desc="Generating FM samples") as progress:
        while num_collected < num_samples:
            current_batch = min(batch_size, num_samples - num_collected)
            x = sampler(current_batch, length, device)
            samples.append(x.cpu())
            num_collected += current_batch
            progress.update(current_batch)
    return torch.cat(samples, dim=0)


def two_point_correlation(x, max_distance=None):
    """Compute periodic C(r) = <x_i x_{i+r}> in real space."""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected x with shape [N, 1, L], got {tuple(x.shape)}")

    length = x.shape[-1]
    if max_distance is None:
        max_distance = length - 1
    if not 0 <= max_distance < length:
        raise ValueError(f"max_distance must be between 0 and {length - 1}")

    return torch.stack(
        [(x * torch.roll(x, shifts=-r, dims=-1)).mean() for r in range(max_distance + 1)]
    )


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
        description="Evaluate periodic real-space two-point correlations for local FM."
    )
    parser.add_argument("--config_L1024", type=str, required=True)
    parser.add_argument("--ckpt_L1024", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=250)
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

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    flags = load_config(args.config_L1024)
    if int(flags.L) != 1024:
        raise ValueError(f"Expected L=1024 in {args.config_L1024}, got L={flags.L}")
    sampler = build_sampler(flags, args.ckpt_L1024, args.model_key, device)

    out_dir = args.out_dir or os.path.join(flags.logdir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    x_data = collect_data_samples(flags, args.num_samples, args.batch_size)
    x_model = collect_generated_samples(
        sampler, flags.L, device, args.num_samples, args.batch_size
    )

    c_data = two_point_correlation(x_data, args.max_distance)
    c_model = two_point_correlation(x_model, args.max_distance)
    csv_path = save_correlation(c_data, c_model, out_dir)

    summary = {
        "num_samples": int(args.num_samples),
        "L": int(flags.L),
        "max_distance": int(len(c_data) - 1),
        "config": args.config_L1024,
        "checkpoint": args.ckpt_L1024,
        "model_key": args.model_key,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Correlation evaluation finished.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
