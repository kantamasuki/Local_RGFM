import copy
import json
import os
import argparse
import warnings
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml_collections import ConfigDict

from dataset import get_dataset
from model import LocalConvDenoiser1D
from rgfm_diffusion import RGDiffusionReal1D
from rgfm_trainer import RGTrainer
from rgfm_sampler import RGSamplerReal, RGSamplerChecker


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()

with open(args.config, "r") as f:
    config_data = json.load(f)
FLAGS = ConfigDict(config_data)

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


def get_chains(batch):
    """
    Accepts either:
      - x
      - (x, label)
      - dict with key "x" or "chain"

    Expected output shape: [B, 1, L_base]
    """
    if isinstance(batch, dict):
        if "x" in batch:
            return batch["x"]
        if "chain" in batch:
            return batch["chain"]
        raise KeyError("dict batch must contain key 'x' or 'chain'")

    if isinstance(batch, (tuple, list)):
        return batch[0]

    return batch


def infiniteloop(dataloader):
    while True:
        for batch in dataloader:
            yield get_chains(batch)


def warmup_lr(step):
    if FLAGS.warmup <= 0:
        return 1.0
    return min(step + 1, FLAGS.warmup) / FLAGS.warmup


def make_dirs():
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, "samples"), exist_ok=True)
    with open(os.path.join(FLAGS.logdir, "saved_config.json"), "w") as f:
        json.dump(config_data, f, indent=4)


def build_model():
    return LocalConvDenoiser1D(
        hidden_dim=FLAGS.hidden_dim,
        time_dim=FLAGS.time_dim,
        depth=FLAGS.depth,
        kernel_size=FLAGS.kernel_size,
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
    dataset = get_dataset(FLAGS.dataset_key, L=FLAGS.base_L)

    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    datalooper = infiniteloop(dataloader)

    net_model = build_model()
    ema_model = copy.deepcopy(net_model).to(device)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    rg_diffusion_real = RGDiffusionReal1D(
        L=FLAGS.base_L,
        sigma_min=FLAGS.sigma_min,
        mu_min=FLAGS.mu_min,
        mL=FLAGS.mL,
        reg_name=FLAGS.reg_name,
    ).to(device)

    t_interval = tuple(FLAGS.t_interval)

    trainer = RGTrainer(
        model=net_model,
        rg_diffusion_real=rg_diffusion_real,
        t_interval=t_interval,
    ).to(device)

    ema_sampler = RGSamplerReal(
        model=ema_model,
        t_interval=t_interval,
        n_total_steps=FLAGS.sample_n_steps,
        method=FLAGS.sample_method,
        rtol=FLAGS.sample_rtol,
        atol=FLAGS.sample_atol,
    ).to(device)

    ema_checker = RGSamplerChecker(
        rg_diffusion_real=rg_diffusion_real,
        sampler=ema_sampler,
        L=FLAGS.L
    ).to(device)

    model_size = sum(p.numel() for p in net_model.parameters())
    print(f"Model params: {model_size / 1e6:.3f} M")
    print(
        f"Training L={FLAGS.L}, base_L={FLAGS.base_L}, "
        f"t_interval={t_interval}, RF={net_model.receptive_field}"
    )

    make_dirs()

    start_step = 0
    for step in tqdm(range(start_step, FLAGS.total_steps)):
        net_model.train()
        optim.zero_grad(set_to_none=True)

        phi_0 = next(datalooper).to(device, non_blocking=True)

        # Expected shape: [B, 1, base_L]
        if phi_0.dim() == 2:
            phi_0 = phi_0.unsqueeze(1)
        assert phi_0.dim() == 3
        assert phi_0.shape[1] == 1
        assert phi_0.shape[-1] == FLAGS.base_L

        loss = trainer(phi_0, resizeL=FLAGS.L)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
        optim.step()
        sched.step()
        ema(net_model, ema_model, FLAGS.ema_decay)


        if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
            ema_model.eval()
            with torch.no_grad():
                x_ref = next(datalooper).to(device, non_blocking=True)
                if x_ref.dim() == 2:
                    x_ref = x_ref.unsqueeze(1)
                x_ref = x_ref[:FLAGS.sample_size]

                x_rec, x_tf = ema_checker(x_ref, return_noisy=True)
                plot_saver(x_rec, os.path.join(FLAGS.logdir, "samples", f"x_rec_{step}.png"))
                plot_saver(x_tf, os.path.join(FLAGS.logdir, "samples", f"x_tf_{step}.png"))

        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
            save_ckpt(
                os.path.join(FLAGS.logdir, "ckpt", f"ckpt_{step // FLAGS.save_step}.pt"),
                net_model,
                ema_model,
                optim,
                sched,
                step,
            )

    # Always save final checkpoint.
    final_step = max(start_step, FLAGS.total_steps) - 1
    save_ckpt(
        os.path.join(FLAGS.logdir, "ckpt", "last.pt"),
        net_model,
        ema_model,
        optim,
        sched,
        final_step,
    )
    

def plot_saver(x, save_path):
    x = x.cpu()
    plt.figure(figsize=(10, 6))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.ylim([-5,5])
        plt.plot(x[i, 0].numpy(), alpha=0.8)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    train()
