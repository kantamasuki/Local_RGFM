"""Generate L=256 images with the complete five-stage local RGFM chain."""

import argparse
import json
import os
from types import SimpleNamespace

import torch
from torchvision.utils import make_grid, save_image

from model import UNet
from patch_maker import PatchMaker
from rgfm_diffusion import RGDiffusionReal
from rgfm_sampler import RGSamplerReal, RGSamplerTotal


def as_namespace(d):
    return SimpleNamespace(**d)


def load_config(path):
    with open(path, "r") as f:
        return as_namespace(json.load(f))


def build_model(FLAGS, device):
    return UNet(
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
        in_channels=5,
        out_channels=3,
    ).to(device)


def load_model_weight(model, ckpt_path, device, key="ema_model"):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if key in ckpt:
            state = ckpt[key]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "net_model" in ckpt:
            state = ckpt["net_model"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state)
    return model


def build_sampler(FLAGS, ckpt_path, device, ckpt_key="ema_model"):
    model = build_model(FLAGS, device)
    load_model_weight(model, ckpt_path, device, key=ckpt_key)
    model.eval()

    patch_maker = PatchMaker(
        L=FLAGS.img_size,
        P=FLAGS.P,
        D=FLAGS.D,
        stride=FLAGS.stride,
        normalize_coord=FLAGS.normalize_coord,
    )

    sampler = RGSamplerReal(
        model=model,
        patch_maker=patch_maker,
        t_interval=tuple(FLAGS.t_interval),
        n_total_steps=FLAGS.sample_n_steps,
        method=FLAGS.sample_method,
        rtol=FLAGS.sample_rtol,
        atol=FLAGS.sample_atol,
    ).to(device)
    sampler.eval()
    return sampler


def build_rg_diffusion(FLAGS, device):
    return RGDiffusionReal(
        L=FLAGS.base_img_size,
        sigma_min=FLAGS.sigma_min,
        mu_min=FLAGS.mu_min,
        m0=FLAGS.m0,
        mL=FLAGS.mL,
        reg_name=FLAGS.reg_name,
    ).to(device)


def save_img(x, path, nrow=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if nrow is None:
        nrow = int(x.shape[0] ** 0.5)
    grid = make_grid((x.clamp(-1, 1) + 1.0) / 2.0, nrow=nrow)
    save_image(grid, path)


def assert_chain(configs):
    # Configs must be ordered large -> small: [L256, L128, L64, L32, L16].
    L_list = [c.img_size for c in configs]
    expected_L_list = [256, 128, 64, 32, 16]
    if L_list != expected_L_list:
        raise ValueError(f"Expected L_list={expected_L_list}, but got {L_list}")

    for i in range(len(L_list) - 1):
        if not (L_list[i] > L_list[i + 1]):
            raise ValueError(f"configs must be ordered large -> small, but got L_list={L_list}")

    for i in range(len(configs) - 1):
        tf_large = float(configs[i].t_interval[1])
        ts_small = float(configs[i + 1].t_interval[0])
        if abs(tf_large - ts_small) > 1e-6:
            raise ValueError(
                "Intervals are not connected: "
                f"L={configs[i].img_size} t_f={tf_large}, "
                f"L={configs[i+1].img_size} t_s={ts_small}. "
                "RGSamplerTotal expects sampler_list[i].t_f == sampler_list[i+1].t_s."
            )


@torch.no_grad()
def sample_total(args):
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # IMPORTANT: order is large -> small.
    configs = [
        load_config(args.config256),
        load_config(args.config128),
        load_config(args.config64),
        load_config(args.config32),
        load_config(args.config16),
    ]
    assert_chain(configs)

    ckpts = [
        args.ckpt256,
        args.ckpt128,
        args.ckpt64,
        args.ckpt32,
        args.ckpt16,
    ]

    samplers = [
        build_sampler(config, ckpt, device, args.ckpt_key)
        for config, ckpt in zip(configs, ckpts)
    ]

    # Use the base RG parameters from the largest-L config.
    rg_diffusion_real = build_rg_diffusion(configs[0], device)

    L_list = [c.img_size for c in configs]
    total_sampler = RGSamplerTotal(
        sampler_list=samplers,
        L_list=L_list,
        rg_diffusion_real=rg_diffusion_real,
    ).to(device)
    total_sampler.eval()

    print("Loaded total sampler")
    print(f"L_list={L_list}")
    for c, ckpt in zip(configs, ckpts):
        print(f"  L={c.img_size}, t_interval={tuple(c.t_interval)}, ckpt={ckpt}")

    sample_id = 0
    all_final = []

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    for b in range(num_batches):
        bs = min(args.batch_size, args.num_samples - sample_id)
        if bs <= 0:
            break

        if args.return_intermediate:
            x0, intermediate = total_sampler(
                batch_size=bs,
                channels=3,
                device=device,
                return_intermediate=True,
            )
        else:
            x0 = total_sampler(
                batch_size=bs,
                channels=3,
                device=device,
                return_intermediate=False,
            )
            intermediate = None

        all_final.append(x0.detach().cpu())

        if b < args.save_batches:
            save_img(x0, os.path.join(args.out_dir, f"sample_batch{b:04d}.png"), nrow=args.nrow)

            if intermediate is not None:
                for j, x in enumerate(intermediate):
                    save_img(
                        x,
                        os.path.join(args.out_dir, f"sample_batch{b:04d}_intermediate{j:02d}_L{x.shape[-1]}.png"),
                        nrow=args.nrow,
                    )

        sample_id += bs

    all_final = torch.cat(all_final, dim=0)
    save_img(all_final[: min(args.grid_size, all_final.shape[0])], os.path.join(args.out_dir, "sample_all_grid.png"), nrow=args.nrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config256", type=str, required=True)
    parser.add_argument("--config128", type=str, required=True)
    parser.add_argument("--config64", type=str, required=True)
    parser.add_argument("--config32", type=str, required=True)
    parser.add_argument("--config16", type=str, required=True)
    parser.add_argument("--ckpt256", type=str, required=True)
    parser.add_argument("--ckpt128", type=str, required=True)
    parser.add_argument("--ckpt64", type=str, required=True)
    parser.add_argument("--ckpt32", type=str, required=True)
    parser.add_argument("--ckpt16", type=str, required=True)
    parser.add_argument("--ckpt_key", type=str, default="ema_model")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--out_dir", type=str, default="./logs/samples")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_batches", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=1)
    parser.add_argument("--nrow", type=int, default=1)
    parser.add_argument("--return_intermediate", action="store_true")
    args = parser.parse_args()
    sample_total(args)


if __name__ == "__main__":
    main()
