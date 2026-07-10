import os

from dotenv import load_dotenv

load_dotenv(".env")

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub


DATASET_NAME = "cifar100-200"


def main() -> None:
    """Load a subset of CIFAR-100 from Hugging Face into local FiftyOne."""
    if DATASET_NAME in fo.list_datasets():
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"Dataset already loaded: {dataset.name} ({len(dataset)} samples)")
        return

    dataset = load_from_hub(
        "uoft-cs/cifar100",
        format="parquet",
        default_media_fields={"filepath": "img"},
        classification_fields=["coarse_label", "fine_label"],
        max_samples=200,
        name=DATASET_NAME,
        persistent=True,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Loaded dataset: {dataset.name} ({len(dataset)} samples)")


if __name__ == "__main__":
    main()
