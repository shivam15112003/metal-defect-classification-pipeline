#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix

# Allow running directly from repo root without installation.
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_classifier.data import discover_paths, resolve_dataset_root  # noqa: E402
from defect_classifier.features import load_split_features  # noqa: E402
from defect_classifier.metadata import build_metadata_feature_map  # noqa: E402
from defect_classifier.model import build_model  # noqa: E402
from defect_classifier.reporting import (  # noqa: E402
    save_classification_report,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_metrics,
    save_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone metal defect classification pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory or zip file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory where predictions and evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=24,
        help="Resize dimension used for grayscale thumbnail features.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=16,
        help="Number of histogram bins for intensity histogram features.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the ExtraTrees classifier.",
    )
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        help="Use metadata.csv when available.",
    )
    parser.add_argument(
        "--prediction-split-name",
        type=str,
        default=None,
        help=(
            "Optional prefix used in predictions.csv paths. "
            "By default, test/ is written as val/ to match the assignment format."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional number of CPU workers for the tree model. Defaults to min(4, cpu_count()).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_dir = output_dir / "_work"
    working_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dataset_root = resolve_dataset_root(Path(args.dataset), working_dir)
    dataset_paths = discover_paths(dataset_root)

    metadata_map, metadata_feature_names = build_metadata_feature_map(dataset_paths.metadata_path)
    use_metadata = args.use_metadata and bool(metadata_map)

    X_train, y_train, _ = load_split_features(
        dataset_paths=dataset_paths,
        split_dir=dataset_paths.train_dir,
        metadata_map=metadata_map,
        use_metadata=use_metadata,
        image_size=args.image_size,
        hist_bins=args.hist_bins,
    )
    X_eval, y_eval, prediction_paths = load_split_features(
        dataset_paths=dataset_paths,
        split_dir=dataset_paths.eval_dir,
        metadata_map=metadata_map,
        use_metadata=use_metadata,
        image_size=args.image_size,
        hist_bins=args.hist_bins,
        prediction_split_name=args.prediction_split_name,
    )
    feature_time = time.time() - t0

    t1 = time.time()
    model = build_model(n_estimators=args.n_estimators, random_state=42, n_jobs=args.n_jobs)
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)
    fit_and_inference_time = time.time() - t1

    accuracy = accuracy_score(y_eval, predictions)
    confusion = confusion_matrix(y_eval, predictions)
    predicted_labels = [dataset_paths.class_names[index] for index in predictions]

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    confusion_png_path = output_dir / "confusion_matrix.png"
    report_csv_path = output_dir / "classification_report.csv"

    save_predictions(prediction_paths, predicted_labels, predictions_path)
    save_confusion_matrix_csv(confusion, dataset_paths.class_names, confusion_csv_path)
    save_confusion_matrix_png(confusion, dataset_paths.class_names, confusion_png_path)
    report = save_classification_report(y_eval, predictions, dataset_paths.class_names, report_csv_path)

    metrics = {
        "accuracy": accuracy,
        "class_names": dataset_paths.class_names,
        "dataset_input": str(Path(args.dataset).resolve()),
        "resolved_dataset_root": str(dataset_paths.root),
        "train_dir": str(dataset_paths.train_dir),
        "eval_dir": str(dataset_paths.eval_dir),
        "prediction_path_prefix": args.prediction_split_name if args.prediction_split_name else ("val" if dataset_paths.eval_dir.name == "test" else dataset_paths.eval_dir.name),
        "used_metadata": use_metadata,
        "metadata_feature_count": len(metadata_feature_names),
        "image_size": args.image_size,
        "hist_bins": args.hist_bins,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs if args.n_jobs is not None else "auto<=4",
        "feature_extraction_seconds": round(feature_time, 4),
        "fit_and_inference_seconds": round(fit_and_inference_time, 4),
        "total_runtime_seconds": round(feature_time + fit_and_inference_time, 4),
        "report": report,
    }
    save_metrics(metrics, metrics_path)

    # Keep the submission folder clean.
    if working_dir.exists():
        shutil.rmtree(working_dir)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Feature extraction time: {feature_time:.2f}s")
    print(f"Fit + inference time: {fit_and_inference_time:.2f}s")
    print(f"Total runtime: {feature_time + fit_and_inference_time:.2f}s")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_png_path}")
    print(f"Saved classification report to: {report_csv_path}")


if __name__ == "__main__":
    main()
