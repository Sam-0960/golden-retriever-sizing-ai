"""
calibrate_from_owner_data.py
------------------------------
Once you've collected real measurements in
data/dog_owner_measurements_template.csv (even 10-20 rows helps), run this
to see how your DERIVED adult neck/chest estimates and the growth curve
compare to reality, and get suggested replacement values.

Run:
    python -m src.calibrate_from_owner_data
"""

import pandas as pd
import os

TEMPLATE_PATH = "data/dog_owner_measurements_template.csv"


def load_owner_data(path=TEMPLATE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(subset=["breed", "age_months"])


def suggest_adult_reference_updates(df, adult_age_threshold=24):
    """
    For dogs at/near adult age (>= adult_age_threshold months), average their
    real neck/chest/back to suggest replacement values for
    breed_adult_reference.csv's currently-DERIVED/blank fields.
    """
    adults = df[df["age_months"] >= adult_age_threshold]
    if adults.empty:
        print(f"No rows with age_months >= {adult_age_threshold} yet -- "
              f"can't suggest adult reference updates until you have some "
              f"fully-grown dogs measured.")
        return None

    summary = adults.groupby(["breed", "sex"])[["neck_cm", "chest_cm", "back_length_cm"]].agg(["mean", "std", "count"])
    print("Suggested adult reference values from your real data:")
    print(summary)
    return summary


def suggest_growth_curve_updates(df):
    """
    For puppies (< 24 months), compare their measured chest_cm against what
    the CURRENT model would have estimated, to see how far off the
    Golden-Retriever-derived curve is when applied to real dogs (especially
    useful for checking the Labrador assumption).
    """
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_adult_reference, load_growth_curve
    from src.measurement_estimator import estimate_measurements

    adult_ref = load_adult_reference()
    growth_curve = load_growth_curve()

    puppies = df[df["age_months"] < 24].dropna(subset=["chest_cm"])
    if puppies.empty:
        print("No puppy rows with chest_cm filled in yet.")
        return None

    rows = []
    for _, row in puppies.iterrows():
        est = estimate_measurements(row["breed"], row["sex"], row["age_months"], row["weight_kg"], adult_ref, growth_curve)
        if est["chest_cm"] is not None:
            rows.append({
                "breed": row["breed"], "age_months": row["age_months"],
                "actual_chest_cm": row["chest_cm"], "model_estimated_chest_cm": round(est["chest_cm"], 1),
                "error_cm": round(row["chest_cm"] - est["chest_cm"], 1),
            })

    comparison = pd.DataFrame(rows)
    print("Model vs. reality comparison:")
    print(comparison)
    return comparison


if __name__ == "__main__":
    df = load_owner_data()
    print(f"Loaded {len(df)} owner-submitted measurement rows.\n")
    suggest_adult_reference_updates(df)
    print()
    suggest_growth_curve_updates(df)
