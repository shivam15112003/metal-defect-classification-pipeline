from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report


def save_predictions(prediction_paths: List[str], predicted_labels: List[str], output_path: Path) -> None:
    pd.DataFrame(
        {
            "image_path": prediction_paths,
            "predicted_label": predicted_labels,
        }
    ).to_csv(output_path, index=False)


def save_confusion_matrix_csv(confusion_matrix_array, class_names: Iterable[str], output_path: Path) -> None:
    pd.DataFrame(confusion_matrix_array, index=class_names, columns=class_names).to_csv(output_path)


def save_confusion_matrix_png(confusion_matrix_array, class_names: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion_matrix_array, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(confusion_matrix_array[i][j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(y_true, y_pred, class_names: List[str], output_path: Path) -> dict:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    pd.DataFrame(report).transpose().to_csv(output_path)
    return report


def save_metrics(metrics: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
