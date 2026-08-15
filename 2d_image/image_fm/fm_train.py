# fm_train.py

import copy
import json
import os
import argparse
import warnings

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from ml_collections import ConfigDict

from dataset import get_dataset
from model import UNet

from patch_maker import PatchMaker
from fm_diffusion import FMDiffusion
from fm_trainer import FMTrainerReal
from fm_sampler import FMSamplerReal, FMReconstructor


# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()

# read config
with open(args.config, "r") as f:
    config_data = json.load(f)
FLAGS = ConfigDict(config_data)

# device
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
else:
    device = torch.device("cpu")


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()

    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay
            + source_dict[key].data * (1.0 - decay)
        )


def get_images(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def infiniteloop(dataloader):
    while True:
        for batch in dataloader:
            yield get_images(batch)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def make_dirs():
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, "samples"), exist_ok=True)

    with open(os.path.join(FLAGS.logdir, "saved_config.json"), "w") as f:
        json.dump(config_data, f, indent=4)


def build_model():
    return UNet(
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
        in_channels=5,
        out_channels=3,
    ).to(device)


def save_ckpt(path, net_model, ema_model, optim, sched, step):
    ckpt = {
        "net_model": net_model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "config": config_data,
    }
    torch.save(ckpt, path)


def train():
    # dataset
    dataset = get_dataset(FLAGS.dataset_key)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    datalooper = infiniteloop(dataloader)

    # model
    net_model = build_model()
    ema_model = copy.deepcopy(net_model).to(device)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # patch maker
    patch_maker = PatchMaker(
        L=FLAGS.img_size,
        P=FLAGS.P,
        D=FLAGS.D,
        stride=FLAGS.stride,
        normalize_coord=FLAGS.normalize_coord,
    )

    # diffusion / trainer
    fm_diffusion = FMDiffusion(
        sigma_min=FLAGS.sigma_min,
    ).to(device)

    trainer = FMTrainerReal(
        model=net_model,
        fm_diffusion=fm_diffusion,
        patch_maker=patch_maker,
    ).to(device)

    # sampler
    ema_sampler = FMSamplerReal(
        model=ema_model,
        patch_maker=patch_maker,
        img_size=FLAGS.img_size,
        n_steps=FLAGS.sample_n_steps,
        method=FLAGS.sample_method,
        rtol=FLAGS.sample_rtol,
        atol=FLAGS.sample_atol,
    ).to(device)

    ema_reconstructor = FMReconstructor(
        fm_diffusion=fm_diffusion,
        fm_sampler=ema_sampler,
    ).to(device)

    # model size
    model_size = sum(p.numel() for p in net_model.parameters())
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # dirs
    make_dirs()

    # training
    for step in tqdm(range(FLAGS.total_steps)):
        net_model.train()

        optim.zero_grad(set_to_none=True)

        x0 = next(datalooper).to(device, non_blocking=True)
        loss = trainer(x0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)

        optim.step()
        sched.step()
        ema(net_model, ema_model, FLAGS.ema_decay)

        # sample
        if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
            ema_model.eval()

            with torch.no_grad():
                # generation: x1 -> x0
                x_gen = ema_sampler(
                    batch_size=FLAGS.sample_size,
                    channels=3,
                    device=device,
                    return_traj=False,
                )

                grid_gen = (make_grid(x_gen) + 1.0) / 2.0
                save_image(
                    grid_gen,
                    os.path.join(FLAGS.logdir, "samples", f"sample_{step}.png"),
                )

                # reconstruction check: x0 -> x_t -> x0
                x_ref = next(datalooper)[:FLAGS.sample_size].to(device)

                x_rec, x_t = ema_reconstructor(
                    x_ref,
                    t_start=FLAGS.reconstruct_t,
                    return_noisy=True,
                )

                grid_rec = (make_grid(x_rec) + 1.0) / 2.0
                grid_t = (make_grid(x_t) + 1.0) / 2.0

                save_image(
                    grid_rec,
                    os.path.join(FLAGS.logdir, "samples", f"rec_{step}.png"),
                )
                save_image(
                    grid_t,
                    os.path.join(FLAGS.logdir, "samples", f"xt_{step}.png"),
                )

        # save
        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
            save_ckpt(
                os.path.join(FLAGS.logdir, "ckpt", f"ckpt_{step // FLAGS.save_step}.pt"),
                net_model,
                ema_model,
                optim,
                sched,
                step,
            )

    final_step = max(FLAGS.total_steps, 1) - 1
    save_ckpt(
        os.path.join(FLAGS.logdir, "ckpt", "last.pt"),
        net_model,
        ema_model,
        optim,
        sched,
        final_step,
    )
if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    train()
