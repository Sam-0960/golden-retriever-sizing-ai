#!/usr/bin/env python3
"""
Dog Virtual Try-On (VTON) — SD 1.5 Inpainting (No IP-Adapter dependency)
Uses: Stable Diffusion 1.5 Inpainting (already cached locally)
Runs: Apple Silicon M4 via PyTorch MPS backend

APPROACH (honest for paper):
  "Garment-conditioned inpainting using Stable Diffusion 1.5. The dog's torso
   region (from YOLO segmentation mask) is inpainted using a detailed textual
   description of the Ruffwear jacket, preserving dog identity, pose, and background.
   Size variation is encoded via prompt engineering (snug/perfect/loose)."

Author: Antigravity VTON pipeline v3 — local SD inpainting (no IP-Adapter)
"""

import os, sys, time, json, numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/Users/udayk/Desktop/CCBD/top10jpgs")
MASK_DIR     = BASE_DIR / "seg_images_hybrid"
GARMENT_PATH = Path("/Users/udayk/.gemini/antigravity/brain/4c03030c-f28a-47bc-a540-f8f938a70622/garment_product_jacket_1785084409380.jpg")
OUTPUT_DIR   = Path("/Users/udayk/Desktop/vton_diffusion_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOG_IDS = ["00002","00003","00004","00005","00006","00007","00008","00009","00010"]

# ─── Size-specific configs & prompt engineering ───────────────────────────────
SIZE_CONFIG = {
    "S": dict(
        prompt_size  = "tight skin-tight short vest dog jacket, fabric pulled flat and taut across upper back, strained straps, short undersized fit, exposing lower back",
        guidance     = 9.0,
        strength     = 0.90,
        num_steps    = 35,
    ),
    "M": dict(
        prompt_size  = "perfectly fitting medium dog jacket, comfortable natural drape from neck to tail base, ideal size",
        guidance     = 7.5,
        strength     = 0.82,
        num_steps    = 35,
    ),
    "L": dict(
        prompt_size  = "loose baggy oversized long dog jacket, sagging fabric folds hanging low over belly and past hips, oversized loose fit",
        guidance     = 8.5,
        strength     = 0.90,
        num_steps    = 35,
    ),
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print(f"[Setup] Device: {DEVICE} | PyTorch: {torch.__version__}")

# ─── Pipeline (cached, no download needed) ────────────────────────────────────
_pipe = None

def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionInpaintPipeline

    print("\n[Pipeline] Loading SD 1.5 Inpainting from cache...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32,   # Use float32 to prevent NaNs on MPS
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing(slice_size=1)
    print("[Pipeline] Ready!\n")
    _pipe = pipe
    return pipe


# ─── Mask: size-dependent torso region ────────────────────────────────────────
def build_torso_mask(mask_path: Path, target_size: tuple, size: str = "M") -> Image.Image:
    """
    Clean binary mask covering torso region, scaled by physical size:
      - Size S (Tight): Shorter length (ends mid-back), tighter chest coverage
      - Size M (Ideal): Standard length (neck to tail base)
      - Size L (Loose): Extended length (covers hips, hangs lower over belly)
    """
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(target_size, Image.LANCZOS)
    m    = np.array(mask)
    h, w = m.shape

    # Threshold → fill holes
    m_bin = (m > 50).astype(np.uint8) * 255
    pil   = Image.fromarray(m_bin)
    pil   = pil.filter(ImageFilter.MaxFilter(size=25))
    m_f   = np.array(pil)

    # Size-dependent boundary thresholds
    if size == "S":
        # Short & tight: clear front 25% (head/neck), clear back 32% (lower back exposed), clear bottom 35% (legs/belly)
        front_cut = int(w * 0.25)
        back_cut  = int(w * 0.68)
        bottom_cut = int(h * 0.65)
    elif size == "L":
        # Loose & long: clear front 18%, extend back to 95% (hip coverage), extend bottom to 78% (low belly hang)
        front_cut = int(w * 0.18)
        back_cut  = int(w * 0.95)
        bottom_cut = int(h * 0.78)
    else:  # "M"
        # Standard ideal fit: neck to tail base
        front_cut = int(w * 0.22)
        back_cut  = int(w * 0.90)
        bottom_cut = int(h * 0.70)

    torso = m_f.copy()
    torso[:, :front_cut] = 0
    torso[:, back_cut:]  = 0
    torso[bottom_cut:, :] = 0
    torso[:int(h * 0.04), :] = 0

    # Smooth edges for seamless blending
    out = Image.fromarray(torso)
    out = out.filter(ImageFilter.GaussianBlur(radius=10))
    out = Image.fromarray(((np.array(out) > 55) * 255).astype(np.uint8))
    return out.convert("L")


# ─── Core generation ──────────────────────────────────────────────────────────
def run_vton_single(pipe, dog_id: str, size: str) -> Path | None:
    out_path = OUTPUT_DIR / f"vton_{dog_id}_{size}.png"
    if out_path.exists():
        print(f"    [SKIP] {out_path.name}")
        return out_path

    dog_path  = BASE_DIR / f"{dog_id}.jpg"
    mask_path = MASK_DIR / f"{dog_id}_hybrid.png"
    if not dog_path.exists() or not mask_path.exists():
        print(f"    [WARN] Missing files for {dog_id}")
        return None

    cfg = SIZE_CONFIG[size]

    # Load + resize to SD native 512×512
    dog_orig   = Image.open(dog_path).convert("RGB")
    orig_size  = dog_orig.size
    SD_SIZE    = (512, 512)

    dog_sd  = dog_orig.resize(SD_SIZE, Image.LANCZOS)
    mask_sd = build_torso_mask(mask_path, SD_SIZE, size=size)

    # Positive prompt — describes jacket with explicit size-dependent physical drape
    pos_prompt = (
        "photo of a dog wearing an olive green army green Ruffwear dog jacket coat, "
        f"{cfg['prompt_size']}, "
        "jacket has black trim velcro belly strap D-ring handle black straps, "
        "realistic photographic quality, natural lighting, fur showing at edges, "
        "photorealistic high detail, 8k photo"
    )
    neg_prompt = (
        "cartoon, painting, illustration, anime, blurry, distorted, "
        "human, human clothes, deformed anatomy, watermark, text, "
        "ugly, low quality, duplicate body parts"
    )

    print(f"    Generating size {size} ({SD_SIZE[0]}px, {cfg['num_steps']} steps)...")
    t0 = time.time()

    with torch.no_grad():
        out = pipe(
            prompt          = pos_prompt,
            negative_prompt = neg_prompt,
            image           = dog_sd,
            mask_image      = mask_sd,
            height          = SD_SIZE[1],
            width           = SD_SIZE[0],
            num_inference_steps = cfg["num_steps"],
            guidance_scale  = cfg["guidance"],
            strength        = cfg["strength"],
            generator       = torch.Generator(device=DEVICE).manual_seed(42 + ord(size)),
        )

    gen_img = out.images[0]

    # Upscale generated back to original resolution
    gen_fullres  = gen_img.resize(orig_size, Image.LANCZOS)
    mask_fullres = build_torso_mask(mask_path, orig_size, size=size)
    # Soften mask for smooth blending at edges
    mask_soft    = mask_fullres.filter(ImageFilter.GaussianBlur(radius=4))

    # Composite: generated torso over original dog image
    final = Image.composite(gen_fullres, dog_orig, mask_soft)
    final.save(out_path, format="PNG")

    elapsed = time.time() - t0
    print(f"    ✓ {out_path.name} — {elapsed:.0f}s ({orig_size[0]}×{orig_size[1]})")

    # Clear MPS memory
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return out_path


# ─── Run ──────────────────────────────────────────────────────────────────────
def run_all(test_dog=None, test_size=None):
    print("=" * 60)
    print("  DOG VTON — SD 1.5 Inpainting (Apple Silicon MPS)")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}\n")

    pipe  = load_pipeline()
    dogs  = [test_dog]  if test_dog  else DOG_IDS
    sizes = [test_size] if test_size else ["S", "M", "L"]
    total = len(dogs) * len(sizes)
    done  = 0
    results = []

    for dog_id in dogs:
        print(f"\n{'─'*50}  Dog {dog_id}  {'─'*10}")
        for size in sizes:
            done += 1
            print(f"\n  [{done}/{total}] {dog_id} size {size}:")
            t0 = time.time()
            try:
                out = run_vton_single(pipe, dog_id, size)
                results.append({"dog": dog_id, "size": size,
                                 "success": out is not None,
                                 "time_s": round(time.time()-t0, 1)})
            except Exception as e:
                print(f"    [ERROR] {e}")
                results.append({"dog": dog_id, "size": size,
                                 "success": False, "error": str(e),
                                 "time_s": round(time.time()-t0, 1)})

    n_ok = sum(r["success"] for r in results)
    print(f"\n{'='*60}")
    print(f"  DONE  {n_ok}/{total} images ✓")
    print(f"  Folder: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dog",  help="Single dog ID e.g. 00006")
    p.add_argument("--size", help="S / M / L")
    p.add_argument("--all",  action="store_true")
    args = p.parse_args()

    if args.all:
        run_all()
    elif args.dog:
        run_all(test_dog=args.dog, test_size=args.size)
    else:
        print("Running test: dog 00006 size M")
        run_all(test_dog="00006", test_size="M")
