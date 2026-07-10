"""Custom tools for the Data Curator ADK agent."""

import os
from pathlib import Path
from typing import Any

_active_fiftyone_session = None


def _dataset_storage_dir(dataset_name: str) -> Path:
    """Get the local storage path for materialized dataset media.

    Args:
        dataset_name: Local FiftyOne dataset name.

    Returns:
        Directory where media files for the dataset should be stored.
    """
    workspace_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )
    datasets_root = Path(
        os.getenv("VISIONOPS_CREW_DATASETS_DIR", workspace_root / "datasets")
    )
    return datasets_root.expanduser() / dataset_name


def _safe_field_name(name: str) -> str:
    """Convert an arbitrary Hugging Face field name into a FiftyOne-safe name.

    Args:
        name: Source field name from a Hugging Face dataset row.

    Returns:
        Field name containing only alphanumeric characters and underscores.
    """
    return "".join(char if char.isalnum() or char == "_" else "_" for char in name)


def _infer_image_field(hf_dataset: Any, requested_field: str) -> str:
    """Infer the image/media column from a Hugging Face dataset.

    Args:
        hf_dataset: Hugging Face dataset split or dataset-like object.
        requested_field: Caller-specified media field. If provided, it is
            returned without inference.

    Returns:
        Name of the inferred image/media column.

    Raises:
        ValueError: If no image-like column can be inferred.
    """
    if requested_field:
        return requested_field

    try:
        from datasets import Image

        for field_name, feature in hf_dataset.features.items():
            if isinstance(feature, Image):
                return field_name
    except Exception:
        pass

    first_row = hf_dataset[0] if len(hf_dataset) else {}
    for field_name, value in first_row.items():
        if hasattr(value, "save"):
            return field_name
        if isinstance(value, dict) and (value.get("path") or value.get("bytes")):
            return field_name

    raise ValueError(
        "Could not infer an image field. Pass media_field, for example "
        "`image`, `img`, or the dataset's image column name."
    )


def _materialize_image(value: Any, output_dir: Path, index: int) -> str:
    """Save an image representation to a local file and return its path.

    Args:
        value: Image value from a Hugging Face row, such as a filepath, dict
            with `path`/`bytes`, or PIL-like object with `save`.
        output_dir: Directory used for newly materialized image files.
        index: Sample index used to create stable generated filenames.

    Returns:
        Existing or newly created image filepath.

    Raises:
        ValueError: If `value` is not a supported image representation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(value, str) and Path(value).exists():
        return value

    if isinstance(value, dict):
        path = value.get("path")
        if path and Path(path).exists():
            return path

        image_bytes = value.get("bytes")
        if image_bytes:
            suffix = Path(path).suffix if path else ".jpg"
            output_path = output_dir / f"{index:06d}{suffix or '.jpg'}"
            output_path.write_bytes(image_bytes)
            return str(output_path)

    if hasattr(value, "save"):
        suffix = Path(getattr(value, "filename", "") or "").suffix or ".png"
        output_path = output_dir / f"{index:06d}{suffix}"
        value.save(output_path)
        return str(output_path)

    raise ValueError(f"Unsupported image value for sample {index}: {type(value)}")


def _load_with_datasets_fallback(
    repo_id: str,
    name: str,
    split: str,
    max_samples: int,
    media_field: str,
    classification_fields: str,
    persistent: bool,
    overwrite: bool,
) -> tuple[Any, bool, dict[str, Any]]:
    """Load from Hugging Face datasets and manually create a FiftyOne dataset.

    Args:
        repo_id: Hugging Face dataset repository identifier.
        name: Local FiftyOne dataset name.
        split: Dataset split to load, or empty to choose a default split.
        max_samples: Maximum number of rows to import. Non-positive values load
            the full selected split.
        media_field: Optional image/media column name.
        classification_fields: Comma-separated fields to import as
            FiftyOne classifications.
        persistent: Whether the created FiftyOne dataset should persist.
        overwrite: Whether to delete an existing dataset with the same name.

    Returns:
        Tuple of the FiftyOne dataset, whether it was newly created, and import
        metadata describing the fallback path.
    """
    import fiftyone as fo
    from datasets import ClassLabel, DatasetDict, load_dataset

    loaded = load_dataset(repo_id, token=os.getenv("HF_TOKEN"))

    selected_split = split
    if isinstance(loaded, DatasetDict):
        if not selected_split:
            selected_split = "train" if "train" in loaded else next(iter(loaded))
        hf_dataset = loaded[selected_split]
    else:
        hf_dataset = loaded
        selected_split = selected_split or None

    if max_samples and max_samples > 0:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    image_field = _infer_image_field(hf_dataset, media_field)
    label_fields = [
        field.strip()
        for field in classification_fields.split(",")
        if field.strip()
    ]

    if name in fo.list_datasets():
        if overwrite:
            fo.delete_dataset(name)
        else:
            return fo.load_dataset(name), False, {
                "importer": "datasets_fallback",
                "split": selected_split,
                "image_field": image_field,
                "label_fields": label_fields,
            }

    dataset = fo.Dataset(name)
    dataset.persistent = persistent

    media_dir = _dataset_storage_dir(name) / "media"
    samples = []
    features = hf_dataset.features

    for index, row in enumerate(hf_dataset):
        filepath = _materialize_image(row[image_field], media_dir, index)
        sample = fo.Sample(filepath=filepath)

        for field_name, value in row.items():
            if field_name == image_field:
                continue

            safe_name = _safe_field_name(field_name)
            feature = features.get(field_name)
            if field_name in label_fields or isinstance(feature, ClassLabel):
                label = (
                    feature.int2str(value)
                    if isinstance(feature, ClassLabel) and value is not None
                    else str(value)
                )
                sample[safe_name] = fo.Classification(label=label)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                sample[safe_name] = value
            else:
                sample[safe_name] = str(value)

        samples.append(sample)

    if samples:
        dataset.add_samples(samples)

    return dataset, True, {
        "importer": "datasets_fallback",
        "split": selected_split,
        "image_field": image_field,
        "label_fields": label_fields,
    }


def _launch_fiftyone_app(
    dataset: Any,
    port: int = 0,
    address: str = "",
    remote: bool = True,
) -> str:
    """Launch the FiftyOne App and keep its session alive in this process.

    Args:
        dataset: FiftyOne dataset to open in the App.
        port: App port. If zero, `FIFTYONE_APP_PORT` or 5151 is used.
        address: Bind address. If empty, `FIFTYONE_APP_ADDRESS` or `0.0.0.0`
            is used.
        remote: Whether to launch the App in remote mode.

    Returns:
        Local browser URL for the launched App session.
    """
    import fiftyone as fo

    app_address = address or os.getenv("FIFTYONE_APP_ADDRESS", "0.0.0.0")
    app_port = port or int(os.getenv("FIFTYONE_APP_PORT", "5151"))

    global _active_fiftyone_session
    _active_fiftyone_session = fo.launch_app(
        dataset=dataset,
        address=app_address,
        port=app_port,
        remote=remote,
        auto=False,
    )
    return f"http://localhost:{_active_fiftyone_session.server_port}"


def load_huggingface_dataset(
    repo_id: str,
    dataset_name: str = "",
    split: str = "",
    max_samples: int = 200,
    format: str = "",
    media_field: str = "",
    classification_fields: str = "",
    persistent: bool = True,
    overwrite: bool = False,
    launch_app: bool = True,
    port: int = 0,
    address: str = "",
    remote: bool = True,
) -> dict[str, Any]:
    """Load a Hugging Face dataset into local FiftyOne.

    The tool first tries FiftyOne's native Hugging Face loader. If the repo does
    not provide FiftyOne metadata, it falls back to `datasets.load_dataset()`,
    materializes image samples under `datasets/<dataset_name>/media`, and
    creates a persistent FiftyOne dataset manually.

    Args:
        repo_id: Hugging Face dataset repository identifier.
        dataset_name: Optional local FiftyOne dataset name.
        split: Optional split to load.
        max_samples: Maximum number of samples to import.
        format: Optional dataset format passed to FiftyOne's hub loader.
        media_field: Optional media column name.
        classification_fields: Comma-separated label fields to import as
            classifications.
        persistent: Whether the local FiftyOne dataset should persist.
        overwrite: Whether to replace an existing dataset with the same name.
        launch_app: Whether to launch the FiftyOne App after loading.
        port: App port. If zero, the environment/default port is used.
        address: App bind address. If empty, the environment/default address is
            used.
        remote: Whether to launch the App in remote mode.

    Returns:
        JSON-compatible dataset metadata and optional App URL.
    """
    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    name = dataset_name or repo_id.replace("/", "-").replace("_", "-").lower()

    if name in fo.list_datasets() and not overwrite:
        dataset = fo.load_dataset(name)
        created = False
        import_info = {"importer": "existing_dataset"}
    else:
        kwargs = {
            "max_samples": max_samples,
            "name": name,
            "persistent": persistent,
            "overwrite": overwrite,
            "token": os.getenv("HF_TOKEN"),
        }
        if split:
            kwargs["split"] = split
        if format:
            kwargs["format"] = format
        if media_field:
            kwargs["default_media_fields"] = {"filepath": media_field}
        if classification_fields:
            kwargs["classification_fields"] = [
                field.strip()
                for field in classification_fields.split(",")
                if field.strip()
            ]

        try:
            dataset = load_from_hub(repo_id, **kwargs)
            created = True
            import_info = {"importer": "fiftyone_hub"}
        except ValueError as exc:
            if "Could not find fiftyone metadata" not in str(exc):
                raise

            dataset, created, import_info = _load_with_datasets_fallback(
                repo_id=repo_id,
                name=name,
                split=split,
                max_samples=max_samples,
                media_field=media_field,
                classification_fields=classification_fields,
                persistent=persistent,
                overwrite=overwrite,
            )

    app_url = (
        _launch_fiftyone_app(
            dataset=dataset,
            port=port,
            address=address,
            remote=remote,
        )
        if launch_app
        else None
    )

    return {
        "created": created,
        "dataset_name": dataset.name,
        "sample_count": len(dataset),
        "media_type": dataset.media_type,
        "persistent": dataset.persistent,
        "app_url": app_url,
        **import_info,
    }


def open_fiftyone_dataset(
    dataset_name: str,
    port: int = 0,
    address: str = "",
    remote: bool = True,
) -> dict[str, Any]:
    """Open an existing local FiftyOne dataset in the App.

    Args:
        dataset_name: Local FiftyOne dataset name to open.
        port: App port. If zero, the environment/default port is used.
        address: App bind address. If empty, the environment/default address is
            used.
        remote: Whether to launch the App in remote mode.

    Returns:
        JSON-compatible status payload with dataset metadata, App URL, or an
        error and available dataset names.
    """
    import fiftyone as fo

    if dataset_name not in fo.list_datasets():
        return {
            "success": False,
            "dataset_name": dataset_name,
            "error": f"Dataset '{dataset_name}' does not exist",
            "available_datasets": fo.list_datasets(),
        }

    dataset = fo.load_dataset(dataset_name)
    app_url = _launch_fiftyone_app(
        dataset=dataset,
        port=port,
        address=address,
        remote=remote,
    )

    return {
        "success": True,
        "dataset_name": dataset.name,
        "sample_count": len(dataset),
        "media_type": dataset.media_type,
        "app_url": app_url,
    }
