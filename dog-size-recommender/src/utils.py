"""
utils.py
--------
Small, dependency-free helpers: unit conversion + input validation.
"""

LB_TO_KG = 0.453592
IN_TO_CM = 2.54

VALID_BREEDS = {"Golden Retriever", "Labrador Retriever"}
VALID_SEXES = {"male", "female"}
VALID_PRODUCT_TYPES = {"clothing", "harness"}


def lbs_to_kg(lbs):
    return lbs * LB_TO_KG


def inches_to_cm(inches):
    return inches * IN_TO_CM


def validate_dog_input(breed, age_months, weight_kg, sex, product_type):
    """
    Raises ValueError with a clear message if anything required is invalid.
    Returns nothing on success.
    """
    errors = []

    if breed not in VALID_BREEDS:
        errors.append(f"breed must be one of {VALID_BREEDS}, got '{breed}'")

    if age_months is None or age_months <= 0:
        errors.append(f"age_months must be positive, got {age_months}")

    if weight_kg is None or weight_kg <= 0:
        errors.append(f"weight_kg must be positive, got {weight_kg}")

    if sex not in VALID_SEXES:
        errors.append(f"sex must be one of {VALID_SEXES}, got '{sex}' (sex is required in this app)")

    if product_type not in VALID_PRODUCT_TYPES:
        errors.append(f"product_type must be one of {VALID_PRODUCT_TYPES}, got '{product_type}'")

    if errors:
        raise ValueError("Invalid input:\n  - " + "\n  - ".join(errors))
