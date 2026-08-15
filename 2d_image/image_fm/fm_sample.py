# fm_sample.py

import json
import argparse
import os
import warnings

import torch
from torchvision.utils import make_grid, save_image
from ml_collections import ConfigDict

from model import UNet
from patch_maker import PatchMaker
from fm_sampler import FMSamplerReal


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./logs/samples")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--nrow", type=int, default=1)

args = parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return ConfigDict(json.load(f))


def get_device(device_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def build_model(FLAGS):
    return UNet(
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
        in_channels=5,
        out_channels=3,
    )


def load_ema_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if "ema_model" in ckpt:
        state = ckpt["ema_model"]
    elif "net_model" in ckpt:
        state = ckpt["net_model"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state)
    return model


@torch.no_grad()
def main():
    warnings.simplefilter(action="ignore", category=FutureWarning)

    FLAGS = load_config(args.config)
    device = get_device(args.device)

    # load model
    model = build_model(FLAGS).to(device)
    model = load_ema_model(model, args.ckpt, device)
    model.eval()

    # patch maker for sampler
    patch_maker = PatchMaker(
        L=FLAGS.img_size,
        P=FLAGS.P,
        D=FLAGS.D,
        stride=FLAGS.stride,
        normalize_coord=FLAGS.normalize_coord,
    )

    # sampler
    sampler = FMSamplerReal(
        model=model,
        patch_maker=patch_maker,
        img_size=FLAGS.img_size,
        n_steps=FLAGS.sample_n_steps,
        method=FLAGS.sample_method,
        rtol=FLAGS.sample_rtol,
        atol=FLAGS.sample_atol,
    ).to(device)


    all_x0 = []
    remaining = args.num_samples
    while remaining > 0:
        cur_bs = min(args.batch_size, remaining)
        
        out = sampler(
            batch_size=cur_bs,
            channels=3,
            device=device,
        )

        all_x0.append(out)

        remaining -= cur_bs

    x0 = torch.cat(all_x0, dim=0)[:args.num_samples]

    os.makedirs(args.out_dir, exist_ok=True)

    x0_path = os.path.join(args.out_dir, 'sample.png')

    grid_x0 = (make_grid(x0, nrow=args.nrow) + 1.0) / 2.0

    save_image(grid_x0, x0_path)

    print(f"saved: {x0_path}")
    print(f"num_samples = {args.num_samples}, chunk batch_size = {args.batch_size}")


if __name__ == "__main__":
    main()
