"""
data_loader.py
---------------
Safe CSV loading with column validation. Every loader here:
  - skips '#' comment lines (our CSVs use these for TODO/source notes)
  - checks required columns exist
  - never crashes on blank cells -- blank stays as NaN, callers decide
    what to do (that's the whole point of "adapt based on available columns")
"""

import os
import pandas as pd

REQUIRED_COLUMNS = {
    "breed_growth_reference.csv": [
        "breed", "sex", "birth_weight_kg", "birth_neck_cm", "birth_chest_cm",
        "birth_back_cm", "growth_complete_months", "source",
    ],
    "breed_adult_reference.csv": [
        "breed", "sex", "adult_weight_min_kg", "adult_weight_max_kg",
        "adult_height_min_cm", "adult_height_max_cm",
        "adult_neck_min_cm", "adult_neck_max_cm",
        "adult_chest_min_cm", "adult_chest_max_cm",
        "adult_back_min_cm", "adult_back_max_cm", "source",
    ],
    "brand_size_chart.csv": [
        "brand", "product_type", "size",
        "neck_min_cm", "neck_max_cm", "chest_min_cm", "chest_max_cm",
        "back_min_cm", "back_max_cm", "weight_min_kg", "weight_max_kg",
        "breed_guideline", "source",
    ],
    "breed_growth_curve.csv": ["breed", "age_months", "fraction_of_adult_chest", "source"],
}


def _load_csv(path, expected_filename):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    required = REQUIRED_COLUMNS.get(expected_filename, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Expected columns: {required}"
        )
    return df


def load_growth_reference(path="data/breed_growth_reference.csv"):
    return _load_csv(path, "breed_growth_reference.csv")


def load_adult_reference(path="data/breed_adult_reference.csv"):
    return _load_csv(path, "breed_adult_reference.csv")


def load_brand_chart(path="data/brand_size_chart.csv"):
    return _load_csv(path, "brand_size_chart.csv")


def load_growth_curve(path="data/breed_growth_curve.csv"):
    return _load_csv(path, "breed_growth_curve.csv")
