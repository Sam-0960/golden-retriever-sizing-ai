"""
measurement_estimator.py
-------------------------
Estimates neck/chest/back when the user hasn't measured their dog directly.
Combines TWO real data sources, each used for what it's actually good for:

1. GIRTH ESTIMATE (primary): data/breed_growth_curve.csv -- REAL Golden
   Retriever chest-girth-by-month data (Russian Retriever Club table),
   reused for Labrador as a flagged assumption. Piecewise-linear
   interpolated, capped at 1.0 (fully adult) at 24 months.

   estimated_X_cm = adult_X_midpoint_cm * fraction_of_adult_chest(age)

   WHY NOT use the birth-weight model for this: weight scales roughly with
   VOLUME (~length^3) while girth is a LINEAR/circumferential measurement.
   Testing confirmed a weight-based fraction badly underestimates early
   girth (e.g. predicted a 1-month-old's chest at ~4cm vs. the real
   documented ~37cm). The real girth-by-age curve doesn't have this problem
   because it's actual observed girth data, not a proxy.

2. WEIGHT PLAUSIBILITY CHECK (secondary): growth_model.py's birth-weight ->
   adult-weight line (real anchors: peer-reviewed Labrador birth weight,
   weaker-but-real Golden Retriever birth weight consensus). Used ONLY to
   flag if the user's entered weight looks off for the dog's stated age --
   not used to compute the girth estimate itself.
"""

import pandas as pd
import numpy as np
from src.growth_model import expected_weight_at_age, get_growth_anchors

ADULT_AGE_MONTHS = 24
WEIGHT_PLAUSIBILITY_TOLERANCE = 0.35  # 35% deviation from expected triggers a flag


def _girth_fraction(breed, age_months, growth_curve):
    rows = growth_curve[growth_curve["breed"] == breed].sort_values("age_months")
    if rows.empty:
        return None
    age_clamped = min(age_months, ADULT_AGE_MONTHS)
    xs = rows["age_months"].values.astype(float)
    ys = rows["fraction_of_adult_chest"].values.astype(float)
    return max(0.0, min(1.0, float(np.interp(age_clamped, xs, ys))))


def _adult_midpoints(breed, sex, adult_reference):
    row = adult_reference[(adult_reference["breed"] == breed) & (adult_reference["sex"] == sex)]
    if row.empty:
        return {"neck": None, "chest": None, "back": None, "weight": None}
    r = row.iloc[0]

    def mid(lo_col, hi_col):
        lo, hi = r.get(lo_col), r.get(hi_col)
        if pd.isna(lo) or pd.isna(hi):
            return None
        return (lo + hi) / 2

    return {
        "neck": mid("adult_neck_min_cm", "adult_neck_max_cm"),
        "chest": mid("adult_chest_min_cm", "adult_chest_max_cm"),
        "back": mid("adult_back_min_cm", "adult_back_max_cm"),
        "weight": mid("adult_weight_min_kg", "adult_weight_max_kg"),
    }


def estimate_measurements(breed, sex, age_months, weight_kg, adult_reference, growth_curve, growth_reference=None):
    """
    Returns:
        {
          "insufficient_data": bool,
          "message": str or None,
          "neck_cm": float or None,
          "chest_cm": float or None,
          "back_length_cm": float or None,
          "is_estimated": {"neck": bool, "chest": bool, "back": bool},
          "growth_fraction": float or None,
          "weight_looks_implausible": bool,
          "expected_weight_kg": float or None,
        }
    """
    fraction = _girth_fraction(breed, age_months, growth_curve)
    if fraction is None:
        return {
            "insufficient_data": True,
            "message": f"No growth curve data available for breed '{breed}'.",
            "neck_cm": None, "chest_cm": None, "back_length_cm": None,
            "is_estimated": {"neck": False, "chest": False, "back": False},
            "growth_fraction": None, "weight_looks_implausible": False, "expected_weight_kg": None,
        }

    adult = _adult_midpoints(breed, sex, adult_reference)

    if all(adult[k] is None for k in ("neck", "chest", "back")):
        return {
            "insufficient_data": True,
            "message": (
                "Insufficient data. Please enter chest girth, neck girth, and "
                f"back length manually -- no adult body-measurement reference "
                f"is available for {breed} ({sex}) in breed_adult_reference.csv."
            ),
            "neck_cm": None, "chest_cm": None, "back_length_cm": None,
            "is_estimated": {"neck": False, "chest": False, "back": False},
            "growth_fraction": round(fraction, 3), "weight_looks_implausible": False, "expected_weight_kg": None,
        }

    weight_flag, expected_wt = False, None
    if growth_reference is not None and adult["weight"] is not None:
        anchors = get_growth_anchors(breed, sex, growth_reference)
        if anchors is not None:
            expected_wt = expected_weight_at_age(
                age_months, anchors["birth_weight_kg"], adult["weight"], anchors["growth_complete_months"]
            )
            if expected_wt:
                weight_flag = abs(weight_kg - expected_wt) / expected_wt > WEIGHT_PLAUSIBILITY_TOLERANCE

    def est(key):
        return adult[key] * fraction if adult[key] is not None else None

    return {
        "insufficient_data": False,
        "message": None,
        "neck_cm": est("neck"),
        "chest_cm": est("chest"),
        "back_length_cm": est("back"),
        "is_estimated": {
            "neck": adult["neck"] is not None,
            "chest": adult["chest"] is not None,
            "back": adult["back"] is not None,
        },
        "growth_fraction": round(fraction, 3),
        "weight_looks_implausible": weight_flag,
        "expected_weight_kg": round(expected_wt, 2) if expected_wt else None,
    }
