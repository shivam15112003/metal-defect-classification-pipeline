# Metal Defect Classification Pipeline

A standalone, production-style image classification pipeline for synthetic industrial metal surface defect detection.

## Problem Summary
The dataset contains grayscale images of metal surfaces with five classes:

- crack
- hole
- normal
- rust
- scratch

The provided dataset layout is:

```text
dataset/
  train/
  test/
  metadata.csv
```

One important dataset detail: `metadata.csv` marks the evaluation split as `val`, while the folder on disk is named `test/`. This repository treats `test/` as the validation split and writes `val/...` paths in `predictions.csv` so the submission matches the task instructions and example format.

## Approach
This solution intentionally favors clarity, correctness, and practical engineering over model complexity.

### Model choice
I used a lightweight `ExtraTreesClassifier` on top of engineered features rather than a heavy CNN. This keeps the solution:

- easy to review
- fast to run on CPU
- reproducible
- strong enough for the assignment

### Features used
The pipeline builds a feature vector from:

1. **Image features**
   - resized grayscale thumbnail pixels
   - intensity histogram
   - simple image statistics (mean, std, min, max, median)

2. **Optional metadata features**
   - width, height
   - base intensity
   - lighting angle
   - noise strength
   - defect count / coverage
   - engineered summaries from defect positions and defect sizes
   - one-hot encoding for texture type

### Practical decisions
- The code works with either a **zip file** or an **already extracted dataset folder**.
- Paths are handled with `pathlib` for cross-platform robustness.
- The model is deterministic with a fixed random seed.
- `generation_seed` and `generation_timestamp` are **intentionally excluded** from metadata features to avoid shortcut learning from synthetic generation artifacts.
- The pipeline saves all evaluation artifacts needed for review.

## Final Result
I ran the pipeline end-to-end and got:

- **Validation accuracy:** `0.9583`

Per-class performance is saved in `artifacts/classification_report.csv`, and the confusion matrix is saved in both CSV and PNG format.

## Repository Structure

```text
metal-defect-classification-pipeline/
  main.py
  requirements.txt
  README.md
  predictions.csv
  artifacts/
    predictions.csv
    metrics.json
    classification_report.csv
    confusion_matrix.csv
    confusion_matrix.png
  src/
    defect_classifier/
      __init__.py
      data.py
      features.py
      metadata.py
      model.py
      reporting.py
```

## How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
From the repository root:

```bash
python main.py --dataset /path/to/dataset-20260324T063048Z-1-001.zip --output-dir artifacts --use-metadata
```

You can also point `--dataset` to an extracted dataset directory instead of the zip.

## Outputs
Running the command above generates:

- `artifacts/predictions.csv`
- `artifacts/metrics.json`
- `artifacts/classification_report.csv`
- `artifacts/confusion_matrix.csv`
- `artifacts/confusion_matrix.png`

A copy of `predictions.csv` is also included at the repository root for convenience.

## Submission Notes
- `predictions.csv` follows the requested format:

```csv
image_path,predicted_label
val/crack/crack_00003.png,crack
```

- Although the dataset folder is named `test/`, the output uses `val/` because the assignment wording and `metadata.csv` identify this as the validation split.

## Runtime
Measured runtime on my run:

- **Feature extraction:** about `26.8s`
- **Fit + inference:** about `4.9s`
- **Total:** about `31.7s`

On a typical laptop, expect roughly **1 to 3 minutes** end-to-end when reading from the zip file, depending on CPU speed and whether dependencies are already installed.

## Assumptions
- The `test/` folder is the validation set referenced in the instructions.
- Using `metadata.csv` is allowed because the assignment explicitly says it is optional to use.
- The evaluation objective is correctness and practicality, not maximum model complexity.
