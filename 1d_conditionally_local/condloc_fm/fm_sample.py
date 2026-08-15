"""Generate samples from a trained local FM model."""

import argparse
import json
import os

import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from fm_sampler import FMSamplerReal
from model import LocalConvDenoiser1D


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
def generate_samples(sampler, length, device, num_samples, batch_size):
    samples = []
    num_generated = 0
    with tqdm(total=num_samples, desc="Generating FM samples") as progress:
        while num_generated < num_samples:
            current_batch = min(batch_size, num_samples - num_generated)
            samples.append(sampler(current_batch, length, device).cpu())
            num_generated += current_batch
            progress.update(current_batch)
    return torch.cat(samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate local FM samples.")
    parser.add_argument("--config_L1024", type=str, required=True)
    parser.add_argument("--ckpt_L1024", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument(
        "--model_key",
        type=str,
        default="ema_model",
        choices=["ema_model", "net_model"],
    )
    args = parser.parse_args()

    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive.")

    flags = load_config(args.config_L1024)
    if int(flags.L) != 1024:
        raise ValueError(f"Expected L=1024 in {args.config_L1024}, got L={flags.L}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    sampler = build_sampler(flags, args.ckpt_L1024, args.model_key, device)
    samples = generate_samples(
        sampler, int(flags.L), device, args.num_samples, args.batch_size
    )

    out_dir = args.out_dir or os.path.join(flags.logdir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "samples.pt")
    torch.save(samples, output_path)

    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
