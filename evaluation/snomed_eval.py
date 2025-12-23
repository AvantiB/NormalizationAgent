# evaluation/snomed_eval.py

import pandas as pd
from typing import Dict


def evaluate_normalization(
    df: pd.DataFrame,
    gold_col: str = "concept_id",
    pred_col: str = "pred_concept_id",
) -> Dict[str, float]:
    """
    Evaluates normalization performance on SNOMED-CT EL dataset.

    Assumes:
      - One row per gold entity
      - Gold concept_id exists
      - Prediction may be null
    """

    total = len(df)

    predicted = df[pred_col]
    correct = df[gold_col].astype(str) == df[pred_col].astype(str)
    # print(correct)

    coverage = None #predicted.mean()
    accuracy = correct[predicted.notnull()].mean()
    strict_accuracy = correct.mean()

    return {
        "total_entities": total,
        "coverage": coverage,
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
    }


def print_eval(metrics: Dict[str, float]):
    print("\n===== SNOMED-CT NORMALIZATION EVAL =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
