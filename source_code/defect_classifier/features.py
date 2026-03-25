from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .data import DatasetPaths, iter_class_images


def extract_image_features(image_path: Path, image_size: int = 24, hist_bins: int = 16) -> np.ndarray:
    """Extract lightweight image features.

    Features:
    - downsampled grayscale thumbnail pixels
    - normalized intensity histogram
    - simple summary statistics
    """
    image = Image.open(image_path).convert("L").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    pixels = arr.reshape(-1).astype(np.float32)
    histogram, _ = np.histogram(arr, bins=hist_bins, range=(0.0, 1.0), density=True)
    histogram = histogram.astype(np.float32)

    stats = np.asarray(
        [
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(np.median(arr)),
        ],
        dtype=np.float32,
    )

    return np.concatenate([pixels, histogram, stats], axis=0)


def load_split_features(
    dataset_paths: DatasetPaths,
    split_dir: Path,
    metadata_map: Dict[str, np.ndarray],
    use_metadata: bool,
    image_size: int,
    hist_bins: int,
    prediction_split_name: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    prediction_paths: List[str] = []

    metadata_dim = len(next(iter(metadata_map.values()))) if use_metadata and metadata_map else 0
    empty_metadata = np.zeros(metadata_dim, dtype=np.float32) if metadata_dim else None

    class_to_index = {name: idx for idx, name in enumerate(dataset_paths.class_names)}
    output_prefix = prediction_split_name if prediction_split_name else ("val" if split_dir.name == "test" else split_dir.name)

    for class_name, image_path in iter_class_images(split_dir, dataset_paths.class_names):
        image_features = extract_image_features(image_path, image_size=image_size, hist_bins=hist_bins)

        if use_metadata and metadata_map:
            metadata_features = metadata_map.get(image_path.name, empty_metadata)
            feature_vector = np.concatenate([image_features, metadata_features], axis=0)
        else:
            feature_vector = image_features

        relative_path = image_path.relative_to(dataset_paths.root)
        path_parts = list(relative_path.parts)
        if path_parts:
            path_parts[0] = output_prefix
        prediction_path = "/".join(path_parts)

        features.append(feature_vector)
        labels.append(class_to_index[class_name])
        prediction_paths.append(prediction_path)

    if not features:
        raise ValueError(f"No images found in split directory: {split_dir}")

    return np.stack(features), np.asarray(labels, dtype=np.int64), prediction_paths
