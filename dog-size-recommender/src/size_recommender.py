"""
size_recommender.py
--------------------
recommend_size(): matches a dog's measurements (actual or estimated) against
brand_size_chart.csv rows for a given product_type, and returns a fit
score (0-100), confidence label, chosen size, and a plain-English explanation.

Scoring philosophy: start at 100, deduct for boundary/out-of-range fit,
deduct for using estimated (not measured) values, deduct for low breed
confidence, deduct for growing puppies. This is a transparent penalty
system, not a black-box model, so every point lost is explainable.
"""

import pandas as pd

CONFIDENCE_HIGH_MIN = 85
CONFIDENCE_MEDIUM_MIN = 65

# Per product_type, how much each measurement should count toward the score.
# Chest girth dominates for both; back length matters more for clothing fit;
# neck is secondary unless the product is a collar (not in scope here).
MEASUREMENT_WEIGHTS = {
    "harness": {"chest": 0.60, "back": 0.15, "neck": 0.25},
    "clothing": {"chest": 0.45, "back": 0.35, "neck": 0.20},
}


def _measurement_fit_score(value, lo, hi):
    """
    Returns (score_0_to_100, status) for one measurement against one size's range.
    status in {"in_range_center", "in_range_edge", "slightly_out", "far_out", "no_data"}
    """
    if value is None or pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return None, "no_data"

    width = hi - lo
    if lo <= value <= hi:
        rel_pos = (value - lo) / width  # 0..1
        dist_from_center = abs(rel_pos - 0.5) * 2  # 0 center .. 1 edge
        if dist_from_center <= 0.6:
            # middle 60% of range -> high fit, scaled 85-100
            score = 100 - (dist_from_center / 0.6) * 15
            return score, "in_range_center"
        else:
            # outer 20% near either boundary -> medium fit, scaled 60-85
            edge_frac = (dist_from_center - 0.6) / 0.4  # 0..1 within the edge band
            score = 85 - edge_frac * 25
            return score, "in_range_edge"

    # outside the range
    overshoot = (lo - value) if value < lo else (value - hi)
    overshoot_pct = overshoot / width
    if overshoot_pct <= 0.05:
        return 45, "slightly_out"      # low but possible fit
    elif overshoot_pct <= 0.10:
        return 20, "far_out"           # poor fit
    else:
        return 0, "far_out"            # reject


def recommend_size(user_input, brand_chart, growth_reference, adult_reference):
    """
    user_input: dict with keys:
        breed, sex, age_months, weight_kg, product_type,
        breed_confidence (0-1),
        neck_cm, chest_cm, back_length_cm (each: float or None),
        measurement_source: {"neck": "actual"|"estimated", "chest": ..., "back": ...}

    Returns:
        {
          "recommended_size": str or None,
          "confidence_score": int (0-100),
          "confidence_label": "High"|"Medium"|"Low",
          "measurements_used": {...},
          "reason": str,
          "warnings": [str, ...],
          "next_best_size": str or None,
          "candidates": [(size, score), ...],
        }
    """
    product_type = user_input["product_type"]
    weights = MEASUREMENT_WEIGHTS[product_type]
    warnings = []

    rows = brand_chart[brand_chart["product_type"] == product_type]
    if rows.empty:
        return {
            "recommended_size": None,
            "confidence_score": 0,
            "confidence_label": "Low",
            "measurements_used": {},
            "reason": f"No brand_size_chart.csv rows found for product_type='{product_type}'.",
            "warnings": ["No confident size found. Please provide manual measurements or check the chart."],
            "next_best_size": None,
            "candidates": [],
        }

    measurements = {
        "chest": user_input.get("chest_cm"),
        "back": user_input.get("back_length_cm"),
        "neck": user_input.get("neck_cm"),
    }
    measurement_source = user_input.get("measurement_source", {})

    candidates = []
    per_size_detail = {}

    for _, row in rows.iterrows():
        size = row["size"]
        weighted_score = 0.0
        total_weight_used = 0.0
        detail = {}

        col_map = {
            "chest": ("chest_min_cm", "chest_max_cm"),
            "back": ("back_min_cm", "back_max_cm"),
            "neck": ("neck_min_cm", "neck_max_cm"),
        }

        for key, (lo_col, hi_col) in col_map.items():
            value = measurements[key]
            score, status = _measurement_fit_score(value, row.get(lo_col), row.get(hi_col))
            detail[key] = {"score": score, "status": status}
            if score is not None:
                w = weights[key]
                weighted_score += score * w
                total_weight_used += w

        if total_weight_used == 0:
            continue  # no usable measurement for this size at all

        size_score = weighted_score / total_weight_used
        candidates.append((size, size_score))
        per_size_detail[size] = detail

    if not candidates:
        return {
            "recommended_size": None,
            "confidence_score": 0,
            "confidence_label": "Low",
            "measurements_used": measurements,
            "reason": "No usable measurements to match against the chart.",
            "warnings": ["No confident size found. Please provide manual measurements."],
            "next_best_size": None,
            "candidates": [],
        }

    candidates.sort(key=lambda c: c[1], reverse=True)
    best_size, best_raw_score = candidates[0]
    next_best_size, next_best_score = (candidates[1] if len(candidates) > 1 else (None, None))

    # "If two sizes are very close, recommend the larger size" -- larger =
    # comes later alphabetically in our size ordering isn't reliable, so we
    # use the chart's row order (already XS -> 6XL) as the size ordering.
    if next_best_size is not None and abs(best_raw_score - next_best_score) < 5:
        size_order = list(rows["size"])
        if size_order.index(next_best_size) > size_order.index(best_size):
            best_size, next_best_size = next_best_size, best_size
            best_raw_score, next_best_score = next_best_score, best_raw_score
        warnings.append(
            f"'{best_size}' and '{next_best_size}' scored very close -- sized up per usual guidance for boundary cases."
        )

    final_score = best_raw_score

    # Penalty: estimated (not measured) values -- only count ones that actually have a value
    n_estimated = sum(
        1 for k in ["neck", "chest", "back"]
        if measurement_source.get(k) == "estimated" and measurements[k] is not None
    )
    if n_estimated > 0:
        final_score -= n_estimated * 8
        warnings.append(f"{n_estimated} of the measurements used were ESTIMATED, not directly measured.")

    # Penalty: low breed classification confidence
    breed_confidence = user_input.get("breed_confidence", 1.0)
    if breed_confidence < 0.6:
        final_score -= 15
        warnings.append("Breed prediction confidence is low -- verify the breed manually.")
    elif breed_confidence < 0.85:
        final_score -= 5

    # Penalty: growth-model flagged the entered weight as implausible for age
    age_months = user_input.get("age_months", 999)
    if user_input.get("weight_looks_implausible"):
        final_score -= 10
        warnings.append(
            f"Entered weight looks unusual for a {age_months}-month-old {user_input.get('breed', 'dog')} "
            f"(expected roughly {user_input.get('expected_weight_kg', '?')}kg) -- "
            "double-check the weight, since it affects the growth-based estimate."
        )

    # Penalty: growing puppy
    if age_months < 12:
        final_score -= 5
        warnings.append("Dog may still grow, consider sizing up if the brand allows returns/exchanges.")

    final_score = max(0, min(100, round(final_score)))

    if final_score >= CONFIDENCE_HIGH_MIN:
        label = "High"
    elif final_score >= CONFIDENCE_MEDIUM_MIN:
        label = "Medium"
    else:
        label = "Low"

    # Detect chest/back disagreement (spec: "explain the conflict")
    detail = per_size_detail.get(best_size, {})
    chest_status = detail.get("chest", {}).get("status")
    back_status = detail.get("back", {}).get("status")
    conflict_note = ""
    if chest_status in ("slightly_out", "far_out") and back_status in ("in_range_center", "in_range_edge"):
        conflict_note = " Note: chest girth and back length point to different sizes -- consider the chest fit as primary."
    elif back_status in ("slightly_out", "far_out") and chest_status in ("in_range_center", "in_range_edge"):
        conflict_note = " Note: back length and chest girth point to different sizes -- consider the chest fit as primary."

    used_desc = {
        k: ("estimated" if measurement_source.get(k) == "estimated" else "actual")
        for k in ["neck", "chest", "back"] if measurements[k] is not None
    }

    reason = (
        f"The dog's {'/'.join(used_desc.keys()) if used_desc else 'measurements'} best fit the '{best_size}' "
        f"range in the {user_input['product_type']} chart (raw fit {best_raw_score:.0f}/100)." + conflict_note
    )

    if label == "Low":
        warnings.append(
            "This is a LOW-CONFIDENCE match -- one or more measurements fall poorly "
            "within this size's range. Manual re-measurement is recommended before purchasing."
        )

    return {
        "recommended_size": best_size,
        "confidence_score": final_score,
        "confidence_label": label,
        "measurements_used": {
            "neck": {"value": measurements["neck"], "source": measurement_source.get("neck")},
            "chest": {"value": measurements["chest"], "source": measurement_source.get("chest")},
            "back": {"value": measurements["back"], "source": measurement_source.get("back")},
        },
        "reason": reason,
        "warnings": warnings,
        "next_best_size": next_best_size,
        "candidates": [(s, round(sc, 1)) for s, sc in candidates],
    }
