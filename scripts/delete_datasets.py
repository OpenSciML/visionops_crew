from dotenv import load_dotenv

load_dotenv(".env")

import fiftyone as fo


def main() -> None:
    """Delete all local FiftyOne datasets."""
    dataset_names = fo.list_datasets()
    if not dataset_names:
        print("No datasets to delete")
        return

    for name in dataset_names:
        fo.delete_dataset(name)
        print(f"Deleted dataset: {name}")


if __name__ == "__main__":
    main()
