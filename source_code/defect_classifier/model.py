from __future__ import annotations

import os

from sklearn.ensemble import ExtraTreesClassifier


def build_model(
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> ExtraTreesClassifier:
    effective_jobs = n_jobs if n_jobs is not None else min(4, os.cpu_count() or 1)
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=effective_jobs,
    )
