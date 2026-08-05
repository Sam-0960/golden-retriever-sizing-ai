"""
growth_model.py
-----------------
The core "how big is this puppy right now" model, per your explicit spec:
linear growth from a REAL birth anchor to a REAL adult anchor, over
growth_complete_months (24, i.e. dogs are treated as fully adult at 2 years),
then CONSTANT after that. NOT a single linear relation across all ages --
piecewise: linear then flat.

Two things get modeled this way:
  1. WEIGHT (birth_weight_kg -> adult_weight_kg): this is the one with real
     anchors at BOTH ends (see data/breed_growth_reference.csv header for
     citations -- Labrador birth weight is peer-reviewed, n=7,827 puppies;
     Golden Retriever birth weight is a weaker breeder/vet-consensus figure).
  2. NECK/CHEST/BACK girth: we do NOT have a real birth-girth anchor for
     either breed (searched specifically, found nothing -- see CSV header).
     So girth estimates are computed as:
         estimated_girth = adult_girth_midpoint * weight_growth_fraction(age)
     i.e. we reuse the WEIGHT curve's shape as a proxy for girth maturity.
     This is a real, documented simplifying assumption (weight-for-age is a
     standard developmental-stage proxy in veterinary growth tracking), but
     it is still an assumption, not a directly-measured girth curve.
"""

import pandas as pd


def weight_growth_fraction(age_months, birth_weight_kg, adult_weight_kg, growth_complete_months=24):
    """
    Fraction of "developmental maturity" reached at this age, based on the
    birth-weight -> adult-weight line. NOT literally "fraction of adult
    weight" past growth_complete_months (returns 1.0, fully mature).

    At age=0 this correctly returns birth_weight/adult_weight (a puppy is
    NOT 0% grown at birth) rather than 0 -- a linear line from a real
    non-zero birth anchor to the adult anchor.
    """
    if adult_weight_kg is None or adult_weight_kg <= 0:
        return None

    if age_months >= growth_complete_months:
        return 1.0

    interpolated_weight = birth_weight_kg + (adult_weight_kg - birth_weight_kg) * (age_months / growth_complete_months)
    fraction = interpolated_weight / adult_weight_kg
    return max(0.0, min(1.0, fraction))


def expected_weight_at_age(age_months, birth_weight_kg, adult_weight_kg, growth_complete_months=24):
    """Same line, but returns an actual expected weight in kg (for the plausibility check)."""
    if age_months >= growth_complete_months:
        return adult_weight_kg
    return birth_weight_kg + (adult_weight_kg - birth_weight_kg) * (age_months / growth_complete_months)


def get_growth_anchors(breed, sex, growth_reference):
    """Returns {"birth_weight_kg": float, "growth_complete_months": int} or None if missing."""
    row = growth_reference[(growth_reference["breed"] == breed) & (growth_reference["sex"] == sex)]
    if row.empty:
        return None
    r = row.iloc[0]
    if pd.isna(r["birth_weight_kg"]):
        return None
    return {
        "birth_weight_kg": float(r["birth_weight_kg"]),
        "growth_complete_months": int(r["growth_complete_months"]) if not pd.isna(r["growth_complete_months"]) else 24,
    }
