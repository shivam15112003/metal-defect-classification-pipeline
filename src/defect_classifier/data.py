from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train_dir: Path
    eval_dir: Path
    metadata_path: Path | None
    class_names: List[str]


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_dataset_root(dataset_path: Path, cache_dir: Path) -> Path:
    """Resolve a dataset path that may be either a directory or a zip file.

    The returned directory must contain at least a train/ folder.
    """
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        extract_dir = cache_dir / "extracted_dataset"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dataset_path, "r") as archive:
            archive.extractall(extract_dir)

        candidates = [
            path
            for path in extract_dir.rglob("*")
            if path.is_dir() and (path / "train").exists()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find an extracted dataset root with a train/ folder inside {dataset_path}."
            )
        return sorted(candidates, key=lambda path: len(path.parts))[0]

    if dataset_path.is_dir():
        if (dataset_path / "train").exists():
            return dataset_path
        candidates = [
            path
            for path in dataset_path.rglob("*")
            if path.is_dir() and (path / "train").exists()
        ]
        if candidates:
            return sorted(candidates, key=lambda path: len(path.parts))[0]

    raise FileNotFoundError(f"Could not resolve dataset root from {dataset_path}.")


def discover_paths(dataset_root: Path) -> DatasetPaths:
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    if (dataset_root / "test").exists():
        eval_dir = dataset_root / "test"
    elif (dataset_root / "val").exists():
        eval_dir = dataset_root / "val"
    else:
        raise FileNotFoundError("Expected either test/ or val/ in the dataset root.")

    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.exists():
        metadata_path = None

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class folders found in {train_dir}")

    return DatasetPaths(
        root=dataset_root,
        train_dir=train_dir,
        eval_dir=eval_dir,
        metadata_path=metadata_path,
        class_names=class_names,
    )


def iter_class_images(split_dir: Path, class_names: Iterable[str]):
    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES and image_path.is_file():
                yield class_name, image_path
