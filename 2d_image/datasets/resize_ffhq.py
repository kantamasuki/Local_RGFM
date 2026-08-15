import argparse
import os

import numpy as np
from cleanfid.resize import make_resizer
from PIL import Image
from tqdm import tqdm

SOURCE_SETTINGS = {
    64: ("thumbnails128x128", None),
    256: ("images1024x1024", 5000),
}


def resize_ffhq(img_size):
    root_dir, max_images = SOURCE_SETTINGS[img_size]
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Source directory '{root_dir}' was not found. "
            "Download the corresponding FFHQ images before running this script."
        )

    fn_resize = make_resizer("PIL", False, "bicubic", (img_size, img_size))
    save_dir = f"ffhq{img_size}"

    source_paths = sorted(
        os.path.join(subdir, filename)
        for subdir, _, filenames in os.walk(root_dir)
        for filename in filenames
        if filename.lower().endswith(".png")
    )
    if max_images is not None:
        source_paths = source_paths[:max_images]

    for source_path in tqdm(source_paths, desc=f"FFHQ{img_size}"):
        filename = os.path.basename(source_path)
        destination_dir = os.path.join(save_dir, f"{filename[:2]}000")
        os.makedirs(destination_dir, exist_ok=True)
        destination_path = os.path.join(destination_dir, filename)

        if os.path.exists(destination_path):
            continue

        with Image.open(source_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        resized = fn_resize(image_array)
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        Image.fromarray(resized).save(destination_path)

    print(f"Processed {len(source_paths)} images from '{root_dir}' into '{save_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Resize FFHQ images.")
    parser.add_argument(
        "--img_size",
        type=int,
        required=True,
        choices=sorted(SOURCE_SETTINGS),
        help="Output resolution. The corresponding FFHQ source directory is selected automatically.",
    )
    args = parser.parse_args()
    resize_ffhq(args.img_size)


if __name__ == "__main__":
    main()
