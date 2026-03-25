#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Config / utilities
# -----------------------------

SEED = 42


@dataclass
class DatasetPaths:
    root: Path
    train_dir: Path
    eval_dir: Path
    metadata_path: Path | None


def resolve_dataset_root(dataset_path: Path, work_dir: Path) -> Path:
    """
    Accepts either:
      - extracted dataset directory
      - zip file containing dataset/
    Returns the resolved dataset root folder that contains train/ and test|val/.
    """
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        extract_dir = work_dir / "_extracted_dataset"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dataset_path, "r") as zf:
            zf.extractall(extract_dir)

        # Handle either:
        #   extract_dir/dataset
        #   or extract_dir/<something>/dataset
        candidates = [p for p in extract_dir.rglob("*") if p.is_dir() and (p / "train").exists()]
        if not candidates:
            raise FileNotFoundError("Could not find extracted dataset root containing a train/ folder.")
        return sorted(candidates, key=lambda p: len(p.parts))[0]

    if dataset_path.is_dir():
        if (dataset_path / "train").exists():
            return dataset_path
        candidates = [p for p in dataset_path.rglob("*") if p.is_dir() and (p / "train").exists()]
        if candidates:
            return sorted(candidates, key=lambda p: len(p.parts))[0]

    raise FileNotFoundError(f"Could not resolve dataset root from: {dataset_path}")


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

    return DatasetPaths(
        root=dataset_root,
        train_dir=train_dir,
        eval_dir=eval_dir,
        metadata_path=metadata_path,
    )


def get_class_names(train_dir: Path) -> List[str]:
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in {train_dir}")
    return class_names


# -----------------------------
# Metadata features
# -----------------------------

def safe_parse_list(value: object) -> list:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    return []


def build_metadata_feature_map(metadata_path: Path | None) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Returns:
      feature_map[filename] -> metadata vector
      metadata_feature_names
    """
    if metadata_path is None:
        return {}, []

    df = pd.read_csv(metadata_path)

    # Avoid obvious generator-only shortcut features.
    # Good practical decision for a submission:
    # do not use generation_timestamp or generation_seed.
    numeric_cols = [
        "width",
        "height",
        "base_intensity",
        "lighting_angle",
        "noise_strength",
        "defect_count",
        "defect_coverage_pct",
    ]
    categorical_cols = ["texture_type"]

    if "defect_positions" in df.columns:
        positions = df["defect_positions"].apply(safe_parse_list)
        df["pos_count"] = positions.apply(len)
        df["pos_x_mean"] = positions.apply(lambda xs: float(np.mean([p[0] for p in xs])) if xs else 0.0)
        df["pos_y_mean"] = positions.apply(lambda xs: float(np.mean([p[1] for p in xs])) if xs else 0.0)
        df["pos_x_std"] = positions.apply(lambda xs: float(np.std([p[0] for p in xs])) if xs else 0.0)
        df["pos_y_std"] = positions.apply(lambda xs: float(np.std([p[1] for p in xs])) if xs else 0.0)
        numeric_cols += ["pos_count", "pos_x_mean", "pos_y_mean", "pos_x_std", "pos_y_std"]

    if "defect_sizes_px" in df.columns:
        sizes = df["defect_sizes_px"].apply(safe_parse_list)
        df["size_count"] = sizes.apply(len)
        df["size_mean"] = sizes.apply(lambda xs: float(np.mean(xs)) if xs else 0.0)
        df["size_std"] = sizes.apply(lambda xs: float(np.std(xs)) if xs else 0.0)
        df["size_max"] = sizes.apply(lambda xs: float(np.max(xs)) if xs else 0.0)
        numeric_cols += ["size_count", "size_mean", "size_std", "size_max"]

    # Normalize numeric cols using train split only if available, otherwise full metadata.
    if "split" in df.columns and (df["split"] == "train").any():
        fit_df = df[df["split"] == "train"].copy()
    else:
        fit_df = df.copy()

    means = fit_df[numeric_cols].mean().fillna(0.0)
    stds = fit_df[numeric_cols].std().replace(0, 1).fillna(1.0)

    texture_values = sorted(df["texture_type"].dropna().astype(str).unique()) if "texture_type" in df.columns else []
    texture_to_idx = {v: i for i, v in enumerate(texture_values)}

    feature_names = numeric_cols + [f"texture_type__{v}" for v in texture_values]
    feature_map: Dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        filename = str(row["filename"])
        values: List[float] = []

        for col in numeric_cols:
            val = row[col] if col in row else 0.0
            if pd.isna(val):
                val = means[col]
            values.append(float((val - means[col]) / stds[col]))

        one_hot = [0.0] * len(texture_values)
        if "texture_type" in row and pd.notna(row["texture_type"]):
            key = str(row["texture_type"])
            if key in texture_to_idx:
                one_hot[texture_to_idx[key]] = 1.0

        feature_map[filename] = np.asarray(values + one_hot, dtype=np.float32)

    return feature_map, feature_names


# -----------------------------
# Image features
# -----------------------------

def extract_image_features(image_path: Path, image_size: int = 32) -> np.ndarray:
    """
    Lightweight, CPU-friendly image features.
    Uses downsampled grayscale pixels + a few summary stats.
    """
    img = Image.open(image_path).convert("L").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0

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

    pixels = arr.reshape(-1).astype(np.float32)
    return np.concatenate([pixels, stats], axis=0)


# -----------------------------
# Dataset assembly
# -----------------------------

def load_split(
    split_dir: Path,
    class_names: List[str],
    dataset_root: Path,
    metadata_map: Dict[str, np.ndarray],
    use_metadata: bool,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    rel_paths: List[str] = []

    empty_meta = None
    if use_metadata:
        meta_dim = len(next(iter(metadata_map.values()))) if metadata_map else 0
        empty_meta = np.zeros(meta_dim, dtype=np.float32)

    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.glob("*.png")):
            img_feat = extract_image_features(image_path, image_size=image_size)

            if use_metadata and metadata_map:
                meta_feat = metadata_map.get(image_path.name, empty_meta)
                feat = np.concatenate([img_feat, meta_feat], axis=0)
            else:
                feat = img_feat

            features.append(feat)
            labels.append(class_idx)
            rel_paths.append(str(image_path.relative_to(dataset_root)).replace("\\", "/"))

    X = np.stack(features)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, rel_paths


# -----------------------------
# Training / evaluation
# -----------------------------

def save_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(Path(args.dataset), output_dir)
    paths = discover_paths(dataset_root)
    class_names = get_class_names(paths.train_dir)

    metadata_map, metadata_feature_names = build_metadata_feature_map(paths.metadata_path)
    use_metadata = args.use_metadata and bool(metadata_map)

    X_train, y_train, _ = load_split(
        split_dir=paths.train_dir,
        class_names=class_names,
        dataset_root=paths.root,
        metadata_map=metadata_map,
        use_metadata=use_metadata,
        image_size=args.image_size,
    )

    X_eval, y_eval, eval_rel_paths = load_split(
        split_dir=paths.eval_dir,
        class_names=class_names,
        dataset_root=paths.root,
        metadata_map=metadata_map,
        use_metadata=use_metadata,
        image_size=args.image_size,
    )

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)

    pred_labels = [class_names[i] for i in pred]
    true_labels = [class_names[i] for i in y_eval]

    accuracy = accuracy_score(y_eval, pred)
    report = classification_report(y_eval, pred, target_names=class_names, output_dict=True, digits=4)
    cm = confusion_matrix(y_eval, pred)

    # predictions.csv
    pred_df = pd.DataFrame(
        {
            "image_path": eval_rel_paths,
            "predicted_label": pred_labels,
        }
    )
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # metrics
    metrics = {
        "accuracy": accuracy,
        "class_names": class_names,
        "report": report,
        "dataset_root": str(paths.root),
        "train_dir": str(paths.train_dir),
        "eval_dir": str(paths.eval_dir),
        "used_metadata": use_metadata,
        "image_size": args.image_size,
        "n_estimators": args.n_estimators,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    save_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix to: {output_dir / 'confusion_matrix.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone defect classification pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory or dataset zip file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for predictions and evaluation artifacts.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Resize dimension used for image features.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the ExtraTrees classifier.",
    )
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        help="Use metadata.csv if available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
