"""Create clean-FID statistics for an ImageFolder dataset."""

import argparse
import json
import os
import random

import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.resize import build_resizer
from torchvision import datasets, transforms
from tqdm import tqdm


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


def make_dataset(data_dir):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return datasets.ImageFolder(root=data_dir, transform=transform)


def get_images(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def to_clean_fid_input(images, resize):
    """Convert images from [-1, 1] to clean-FID's 299 x 299 input."""
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


@torch.no_grad()
def collect_features(
    dataloader,
    feature_model,
    resize,
    device,
    num_images,
):
    features = []
    num_collected = 0

    for batch in tqdm(dataloader, desc="Extracting clean-FID features"):
        remaining = num_images - num_collected
        if remaining <= 0:
            break

        images = get_images(batch).float()
        if images.shape[0] > remaining:
            images = images[:remaining]

        fid_input = to_clean_fid_input(images, resize).to(
            device, non_blocking=True
        )
        features.append(feature_model(fid_input).detach().cpu().numpy())
        num_collected += images.shape[0]

    if not features:
        raise RuntimeError("No features were collected from the dataset")
    return np.concatenate(features, axis=0)


def save_stats(path, features, metadata, save_features):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "mu": np.mean(features, axis=0),
        "sigma": np.cov(features, rowvar=False),
        "metadata_json": json.dumps(metadata, indent=2, sort_keys=True),
    }
    if save_features:
        payload["features"] = features
    np.savez(path, **payload)
    print(f"Saved clean-FID statistics to {path}")


def default_output_path(data_dir, image_size):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    filename = f"{dataset_name}_raw_L{image_size}_cleanfid.npz"
    return os.path.join(project_dir, "FID_features", filename)


def main():
    parser = argparse.ArgumentParser(
        description="Create clean-FID statistics for original dataset images."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="ImageFolder-compatible dataset directory.",
    )
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Use a shuffled subset when num_images is smaller than the dataset.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output .npz path. A path under 2d_image/FID_features is used by default.",
    )
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Also store individual Inception features in the output file.",
    )
    args = parser.parse_args()

    if args.num_images <= 1:
        raise ValueError("num_images must be greater than 1")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = make_dataset(args.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {args.data_dir}")

    first_image, _ = dataset[0]
    image_size = int(first_image.shape[-1])
    num_images = min(args.num_images, len(dataset))
    if num_images < args.num_images:
        print(
            f"Requested {args.num_images} images, but the dataset contains "
            f"{len(dataset)}. Using all {num_images} images."
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset: {os.path.abspath(args.data_dir)}")
    print(f"Images used: {num_images}")
    print(f"Device: {device}")
    print("Loading clean-FID feature extractor...")
    feature_model = build_feature_extractor(mode="clean", device=device)
    feature_model.eval()
    resize = build_resizer(mode="clean")

    features = collect_features(
        dataloader=dataloader,
        feature_model=feature_model,
        resize=resize,
        device=device,
        num_images=num_images,
    )

    output_path = args.out_path or default_output_path(
        args.data_dir, image_size
    )
    metadata = {
        "data_dir": os.path.abspath(args.data_dir),
        "num_images": int(features.shape[0]),
        "image_size": image_size,
        "cleanfid_mode": "clean",
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
    }
    save_stats(
        output_path,
        features,
        metadata,
        save_features=args.save_features,
    )


if __name__ == "__main__":
    main()
