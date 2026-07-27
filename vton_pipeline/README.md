# Canine Virtual Try-On (VTON) Pipeline

This directory contains the complete **diffusion-based Virtual Try-On (VTON) pipeline** for dog apparel sizing.

## Directory Structure

- `vton_diffusion.py`: Main Stable Diffusion 1.5 inpainting script with size-dependent mask scaling and MPS hardware acceleration.
- `fit_comparison.py`: 3D mesh extent measurement and S/M/L size recommendation engine.
- `garment_product_jacket.jpg`: Reference garment image (olive green Ruffwear jacket).
- `top10jpgs/`: Benchmark dog photos (`00002.jpg` - `00010.jpg`) and hybrid torso segmentation masks (`seg_images_hybrid/`).
- `vton_diffusion_results/`: 27 generated photorealistic VTON benchmark images (9 dogs × 3 sizes: S, M, L).

## How to Run

### Requirements
```bash
pip install torch diffusers transformers accelerate Pillow numpy
```

### Execution
Run VTON generation locally (MPS on Apple Silicon / CUDA on NVIDIA / CPU):

```bash
# Run test on a single dog (e.g. dog 00006)
python3 vton_diffusion.py --dog 00006

# Run full batch for all 9 benchmark dogs x 3 sizes (27 images)
python3 vton_diffusion.py --all
```
