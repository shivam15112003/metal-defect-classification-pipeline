from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NUMERIC_FEATURE_COLUMNS = [
    "width",
    "height",
    "base_intensity",
    "lighting_angle",
    "noise_strength",
    "defect_count",
    "defect_coverage_pct",
]

# These fields are intentionally excluded to avoid shortcut learning.
EXCLUDED_COLUMNS = {"generation_seed", "generation_timestamp"}


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
        except (ValueError, SyntaxError):
            return []
    return []


def build_metadata_feature_map(metadata_path: Path | None) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if metadata_path is None:
        return {}, []

    df = pd.read_csv(metadata_path)

    numeric_cols = list(NUMERIC_FEATURE_COLUMNS)

    if "defect_positions" in df.columns:
        positions = df["defect_positions"].apply(safe_parse_list)
        df["pos_count"] = positions.apply(len)
        df["pos_x_mean"] = positions.apply(
            lambda xs: float(np.mean([point[0] for point in xs])) if xs else 0.0
        )
        df["pos_y_mean"] = positions.apply(
            lambda xs: float(np.mean([point[1] for point in xs])) if xs else 0.0
        )
        df["pos_x_std"] = positions.apply(
            lambda xs: float(np.std([point[0] for point in xs])) if xs else 0.0
        )
        df["pos_y_std"] = positions.apply(
            lambda xs: float(np.std([point[1] for point in xs])) if xs else 0.0
        )
        numeric_cols += ["pos_count", "pos_x_mean", "pos_y_mean", "pos_x_std", "pos_y_std"]

    if "defect_sizes_px" in df.columns:
        sizes = df["defect_sizes_px"].apply(safe_parse_list)
        df["size_count"] = sizes.apply(len)
        df["size_mean"] = sizes.apply(lambda xs: float(np.mean(xs)) if xs else 0.0)
        df["size_std"] = sizes.apply(lambda xs: float(np.std(xs)) if xs else 0.0)
        df["size_max"] = sizes.apply(lambda xs: float(np.max(xs)) if xs else 0.0)
        numeric_cols += ["size_count", "size_mean", "size_std", "size_max"]

    fit_df = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    means = fit_df[numeric_cols].mean().fillna(0.0)
    stds = fit_df[numeric_cols].std().replace(0, 1).fillna(1.0)

    texture_values = (
        sorted(df["texture_type"].dropna().astype(str).unique()) if "texture_type" in df.columns else []
    )
    texture_to_index = {value: index for index, value in enumerate(texture_values)}

    feature_names = list(numeric_cols) + [f"texture_type__{value}" for value in texture_values]
    feature_map: Dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        values: List[float] = []
        for col in numeric_cols:
            val = row[col] if col in row else 0.0
            if pd.isna(val):
                val = means[col]
            values.append(float((val - means[col]) / stds[col]))

        one_hot = [0.0] * len(texture_values)
        if "texture_type" in row and pd.notna(row["texture_type"]):
            key = str(row["texture_type"])
            if key in texture_to_index:
                one_hot[texture_to_index[key]] = 1.0

        feature_map[str(row["filename"])] = np.asarray(values + one_hot, dtype=np.float32)

    return feature_map, feature_names
