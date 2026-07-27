"""
FIT COMPARISON MODULE
======================
Takes the calibrated measurements already produced by the existing
TripoSR + trimesh pipeline, and produces two outputs:

  1. A quantified fit-comparison chart (bar/gauge style) showing how
     Small / Medium / Large sizes of a chosen garment would fit the
     measured dog, based on real percentage differences -- not guesses.

  2. A scaled visual overlay: the dog's photo with a garment outline
     scaled to S / M / L, placed side-by-side, so the tightness/looseness
     is visually obvious and still traceable back to real numbers.

This does NOT use any generative model. Every number and every
visual scale factor comes directly from the measurements already
computed earlier in the pipeline (chest_girth, body_length, height).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# ---------------------------------------------------------------
# STEP 1: define the garment size chart
# ---------------------------------------------------------------
# Real size chart pulled from Ruffwear dog jackets/harnesses:
# Brand: Ruffwear
# URL: https://ruffwear.com/pages/dog-girth-size-chart
# Values represent the midpoints of the published chest girth fit ranges:
# - XXS (33 - 43 cm) -> Midpoint: 38.0 cm
# - XS (43 - 56 cm)  -> Midpoint: 49.5 cm
# - S (56 - 69 cm)   -> Midpoint: 62.5 cm
# - M (69 - 81 cm)   -> Midpoint: 75.0 cm
# - L (81 - 91 cm)   -> Midpoint: 86.0 cm
# - XL (91 - 107 cm) -> Midpoint: 99.0 cm

GARMENT_SIZE_CHART = {
    "XXS": {"chest_cm": 38.0},
    "XS": {"chest_cm": 49.5},
    "S": {"chest_cm": 62.5},
    "M": {"chest_cm": 75.0},
    "L": {"chest_cm": 86.0},
    "XL": {"chest_cm": 99.0},
}

# Fit tolerance band: within +/- this percent is considered "good fit"
GOOD_FIT_TOLERANCE_PCT = 8


# ---------------------------------------------------------------
# STEP 2: compute fit verdicts for each size
# ---------------------------------------------------------------
def compute_fit_verdicts(measured_chest_girth_cm, size_chart=GARMENT_SIZE_CHART):
    """
    measured_chest_girth_cm: the dog's actual measured chest girth,
    coming directly from the existing measure_mesh.py output.

    Returns a dict per size: percent difference and a verdict label.
    """
    results = {}
    for size_label, spec in size_chart.items():
        garment_chest = spec["chest_cm"]
        pct_diff = ((garment_chest - measured_chest_girth_cm) / measured_chest_girth_cm) * 100

        if abs(pct_diff) <= GOOD_FIT_TOLERANCE_PCT:
            verdict = "Good fit"
        elif pct_diff < -GOOD_FIT_TOLERANCE_PCT:
            verdict = "Tight"
        else:
            verdict = "Loose"

        results[size_label] = {
            "garment_chest_cm": garment_chest,
            "pct_diff": pct_diff,
            "verdict": verdict,
        }
    return results


# ---------------------------------------------------------------
# STEP 3: quantified fit chart (Option 1 -- the scientific result)
# ---------------------------------------------------------------
def plot_fit_chart(measured_chest_girth_cm, dog_name="Test Dog", save_path="fit_chart.png"):
    verdicts = compute_fit_verdicts(measured_chest_girth_cm)

    sizes = list(verdicts.keys())
    pct_diffs = [verdicts[s]["pct_diff"] for s in sizes]
    colors = []
    for s in sizes:
        v = verdicts[s]["verdict"]
        if v == "Good fit":
            colors.append("#2E8B57")   # green
        elif v == "Tight":
            colors.append("#C0392B")   # red
        else:
            colors.append("#D68910")   # amber

    fig, ax = plt.subplots(figsize=(8, 5))

    # shaded "good fit" zone
    ax.axhspan(-GOOD_FIT_TOLERANCE_PCT, GOOD_FIT_TOLERANCE_PCT,
               color="#2E8B57", alpha=0.12, label="Good fit zone")

    bars = ax.bar(sizes, pct_diffs, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Garment chest vs. measured chest girth (%)", fontsize=11)
    ax.set_title(f"Predicted Fit by Size -- {dog_name}\n(measured chest girth: {measured_chest_girth_cm:.1f} cm)",
                 fontsize=13, fontweight="bold")

    for bar, s in zip(bars, sizes):
        v = verdicts[s]
        label = f"{v['verdict']}\n({v['pct_diff']:+.1f}%)"
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + (2 if y >= 0 else -2),
                label, ha="center", va="bottom" if y >= 0 else "top", fontsize=10, fontweight="bold")

    ax.set_ylim(min(pct_diffs) - 15, max(pct_diffs) + 15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved fit chart to {save_path}")
    return verdicts


# ---------------------------------------------------------------
# STEP 4: scaled visual overlay (Option 2 -- the demoable visual)
# ---------------------------------------------------------------
def plot_scaled_overlay(dog_photo_path, measured_chest_girth_cm,
                          garment_outline_width_px=200, garment_outline_height_px=140,
                          save_path="fit_overlay.png"):
    """
    dog_photo_path: path to the original 2D dog photo (same one fed into TripoSR)
    Draws three panels side by side: Small / Medium / Large, each with a
    simple rectangular garment-outline scaled by the real ratio between
    that size's chest spec and the dog's actual measured girth.

    This is basic image compositing -- NOT a generative model. The
    garment shape is a placeholder rectangle representing the chest
    opening; swap in a real traced garment silhouette PNG for a nicer
    look, keeping the same scaling logic.
    """
    verdicts = compute_fit_verdicts(measured_chest_girth_cm)
    dog_img = Image.open(dog_photo_path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # We plot the S, M, L panels specifically (or the first 3 if they exist in results)
    # The prompt explicitly specifies S / M / L
    target_sizes = ["S", "M", "L"]
    
    for ax, size_label in zip(axes, target_sizes):
        v = verdicts[size_label]
        ax.imshow(dog_img)
        ax.axis("off")

        # scale factor: how much bigger/smaller the garment opening is
        # relative to the dog's actual chest girth
        scale_factor = v["garment_chest_cm"] / measured_chest_girth_cm

        w, h = dog_img.size
        box_w = garment_outline_width_px * scale_factor
        box_h = garment_outline_height_px * scale_factor

        # center the outline roughly over the torso region of the photo
        cx, cy = w / 2, h * 0.55

        rect = patches.FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=6,rounding_size=15",
            linewidth=3, edgecolor="#C0392B" if v["verdict"] == "Tight"
                        else "#2E8B57" if v["verdict"] == "Good fit"
                        else "#D68910",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_title(f"Size {size_label} -- {v['verdict']} ({v['pct_diff']:+.1f}%)",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved scaled overlay to {save_path}")


# ---------------------------------------------------------------
# EXAMPLE USAGE -- using the real measured value from today's test run
# ---------------------------------------------------------------
if __name__ == "__main__":
    MEASURED_CHEST_GIRTH_CM = 114.3   # from the actual pipeline output

    verdicts = plot_fit_chart(MEASURED_CHEST_GIRTH_CM, dog_name="Golden Retriever Test Case")
    print(verdicts)

    # Uncomment once you have the real dog photo path available:
    # plot_scaled_overlay("dog.jpg", MEASURED_CHEST_GIRTH_CM)
