# dataset.py
from torchvision import datasets, transforms


def get_dataset(dataset_key):
    if dataset_key == "ffhq64":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        dataset = datasets.ImageFolder(
            root="PATH_TO_FFHQ64_DATASET",
            transform=transform,
        )

        return dataset

    if dataset_key == "ffhq256":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        dataset = datasets.ImageFolder(
            root="PATH_TO_FFHQ256_DATASET",
            transform=transform,
        )

        return dataset

    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
