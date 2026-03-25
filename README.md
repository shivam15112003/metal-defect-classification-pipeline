# Metal Defect Classification Pipeline

A clean, standalone image classification pipeline for the Internship 2026 computer vision assignment.

## Overview
This project trains a lightweight classifier on labeled grayscale defect images, runs inference on the validation split, evaluates model performance, and generates `predictions.csv` in the required format.

The dataset contains five classes:
- `crack`
- `hole`
- `normal`
- `rust`
- `scratch`

## Repository Structure

```text
metal-defect-classification-pipeline/
  main.py
  requirements.txt
  README.md
  .gitignore
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

## How to Run the Code

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the pipeline
From the repository root:

```bash
python main.py --dataset /path/to/dataset-20260324T063048Z-1-001.zip --output-dir artifacts --use-metadata
```

You can also pass an extracted dataset directory instead of the zip file.

## Outputs Generated
Running the command above creates:
- `artifacts/predictions.csv`
- `artifacts/metrics.json`
- `artifacts/classification_report.csv`
- `artifacts/confusion_matrix.csv`
- `artifacts/confusion_matrix.png`

A copy of `predictions.csv` is also kept at the repository root for convenience.

## Approach
This solution prioritizes code quality, clarity, correctness, and practical engineering decisions over model complexity.

### Model choice
I used an `ExtraTreesClassifier` instead of a heavy CNN. This makes the solution:
- simple to review
- fast on CPU
- deterministic and reproducible
- strong enough for the task without unnecessary complexity

### Feature design
The model uses a combination of lightweight engineered features:

#### Image features
- resized grayscale thumbnail pixels
- normalized intensity histogram
- simple image statistics such as mean, standard deviation, min, max, and median

#### Optional metadata features
When `--use-metadata` is enabled and `metadata.csv` is present, the pipeline also uses:
- width and height
- base intensity
- lighting angle
- noise strength
- defect count and defect coverage percentage
- aggregated summaries from defect positions
- aggregated summaries from defect sizes
- one-hot encoding of texture type

### Practical engineering decisions
- Works with both zipped and extracted datasets
- Uses `pathlib` for clean cross-platform paths
- Keeps the project modular by separating data loading, feature extraction, modeling, metadata handling, and reporting
- Fixes the random seed for reproducibility
- Excludes `generation_seed` and `generation_timestamp` from training features to avoid shortcut learning from synthetic generation artifacts

## Validation / Evaluation
The code evaluates the model on the validation split and reports:
- accuracy
- confusion matrix
- per-class precision, recall, and F1 score

### Final result from the packaged run
- **Validation accuracy:** `0.9583`

The full per-class report is saved in `artifacts/classification_report.csv`.

## Assumptions Made
- The folder named `test/` is the validation split referenced by the assignment, because `metadata.csv` marks those samples with `split=val`.
- Therefore, the pipeline evaluates on the on-disk `test/` folder but writes `val/...` paths in `predictions.csv` so the output matches the task wording and expected format.
- Using `metadata.csv` is allowed because the assignment does not forbid auxiliary tabular information, and the approach remains practical and transparent.

## Notes
- The submission intentionally favors a strong, reviewable baseline over a more complex deep learning system.
- The project is designed to be easy to explain during a follow-up interview.
