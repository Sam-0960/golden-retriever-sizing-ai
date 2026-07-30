# Canine Virtual Try-On (VTON) Pipeline

This directory contains the complete **diffusion-based Virtual Try-On (VTON) pipeline** for dog apparel sizing.
It includes two pipeline variants: a **baseline SD inpainting** pipeline and an upgraded **ControlNet depth-guided** pipeline.

---

## Directory Structure

### Baseline SD Inpainting Pipeline
- `vton_diffusion.py` — Main Stable Diffusion 1.5 inpainting script with size-dependent mask scaling and MPS hardware acceleration.
- `fit_comparison.py` — 3D mesh extent measurement and S/M/L size recommendation engine.
- `garment_product_jacket.jpg` — Reference garment image (olive green Ruffwear jacket).
- `top10jpgs/` — Benchmark dog photos (`00002.jpg`–`00010.jpg`) and hybrid torso segmentation masks (`seg_images_hybrid/`).
- `vton_diffusion_results/` — 27 generated photorealistic VTON benchmark images (9 dogs × 3 sizes: S, M, L).

### ControlNet Depth-Guided Pipeline *(Upgraded)*
- `vton_controlnet.py` — Upgraded VTON script using ControlNet depth conditioning + MiDaS depth estimation.
- `screen_best_images.py` — YOLO-based image quality screener that scores images by aspect ratio and coverage, selecting the best side-profile dog photos.
- `labrador_top20.json` — Screening results: top 20 scored labrador images selected for ControlNet inference.
- `vton_controlnet_results/` — 20 ControlNet-generated VTON images (best Labrador dataset images) + their depth maps.

---

## How to Run

### Requirements
```bash
pip install torch diffusers transformers accelerate Pillow numpy ultralytics
```

### Baseline SD Inpainting
Run VTON generation locally (MPS on Apple Silicon / CUDA on NVIDIA / CPU):

```bash
# Run test on a single dog (e.g. dog 00006)
python3 vton_diffusion.py --dog 00006

# Run full batch for all 9 benchmark dogs x 3 sizes (27 images)
python3 vton_diffusion.py --all
```

### ControlNet Depth-Guided VTON *(Recommended)*
Uses MiDaS depth estimation + ControlNet to make the jacket follow the real 3D body curve:

```bash
# Run on a single image
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 vton_controlnet.py --image path/to/dog.jpg

# Run on best 20 labrador images (pre-screened for side-profile quality)
python3 screen_best_images.py        # generates labrador_top20.json
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run_top20_controlnet.py
```

> **Note**: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` when running on Apple Silicon.

---

## Pipeline Architecture

```
Dog Photo
    │
    ├── YOLOv8-seg ──────────────────► Torso mask (inpainting region)
    │
    ├── MiDaS DPT-Large ─────────────► Depth map (3D body geometry)
    │                                         │
    │                                         ▼
    └── Stable Diffusion 1.5 Inpainting + ControlNet Depth
                    │
                    ▼
          VTON output (jacket draped along real 3D body contour)
```

## Pipeline Comparison (Ablation)

| Method | Images | Key Difference |
|---|---|---|
| Baseline SD Inpainting | 27 (S/M/L sizes) | Text + mask only, no geometry awareness |
| ControlNet Depth-Guided | 20 (best screened) | Depth map conditions jacket shape on real 3D body curve |
