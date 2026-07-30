#!/usr/bin/env python3
"""
ControlNet Depth-Guided Dog VTON Pipeline

How it works:
  1. Estimate a depth map from the dog photo using MiDaS (DPT-Large)
  2. Use the depth map as ControlNet conditioning → jacket follows real 3D body curve
  3. Run Stable Diffusion Inpainting + ControlNet depth → VTON output

Why depth matters:
  - Standard SD inpainting fills the mask region without knowing body shape
  - ControlNet depth tells SD exactly how the dog's back curves in 3D space
  - Result: jacket drapes realistically along the body contour

Author: Antigravity VTON pipeline
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Setup] Device: {DEVICE} | PyTorch: {torch.__version__}")

BASE_DIR   = Path("/Users/udayk/Desktop/CCBD")
OUTPUT_DIR = BASE_DIR / "vton_controlnet_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Step 1: Depth Estimation (MiDaS via HuggingFace DPT) ────────────────────
_depth_pipe = None
def get_depth_map(image: Image.Image, target_size=(512, 512)) -> Image.Image:
    """Estimate depth from a single RGB image using Intel DPT-Large (MiDaS)."""
    global _depth_pipe
    if _depth_pipe is None:
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        print("[Depth] Loading MiDaS DPT-Large depth estimator...")
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        model.eval()
        _depth_pipe = (processor, model)
        print("[Depth] Depth model loaded!\n")

    processor, model = _depth_pipe
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # shape: [1, H, W]

    # Normalize to 0–255
    depth = predicted_depth.squeeze().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = ((depth - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)

    depth_pil = Image.fromarray(depth_norm).convert("RGB")
    depth_pil = depth_pil.resize(target_size, Image.LANCZOS)
    return depth_pil


# ─── Step 2: Dog Torso Mask (YOLOv8-seg) ─────────────────────────────────────
_yolo = None
def get_torso_mask(img_path: Path, target_size=(512, 512)) -> Image.Image:
    """YOLOv8 segmentation → torso-only mask."""
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        yolo_path = BASE_DIR / "yolov8n-seg.pt"
        if not yolo_path.exists():
            print("[YOLO] Downloading yolov8n-seg.pt...")
        _yolo = YOLO(str(yolo_path) if yolo_path.exists() else "yolov8n-seg.pt")

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    results = _yolo(img_path, verbose=False)
    mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if results and results[0].masks is not None:
        boxes = results[0].boxes
        masks = results[0].masks
        dog_idx = next((i for i, c in enumerate(boxes.cls) if int(c) == 16), 0)
        m = masks.data[dog_idx].cpu().numpy()
        m_pil = Image.fromarray((m * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.LANCZOS)
        mask_np = np.array(m_pil)
    else:
        print("  [WARN] No dog segmentation found — using center heuristic")
        mask_np[int(orig_h*0.2):int(orig_h*0.7), int(orig_w*0.2):int(orig_w*0.8)] = 255

    m_bin = (mask_np > 50).astype(np.uint8) * 255
    pil_m = Image.fromarray(m_bin).resize(target_size, Image.LANCZOS)
    pil_m = pil_m.filter(ImageFilter.MaxFilter(size=21))
    m_f = np.array(pil_m)
    h, w = m_f.shape

    # Crop to torso (remove head, tail, legs)
    torso = m_f.copy()
    torso[:, :int(w * 0.22)] = 0   # head
    torso[:, int(w * 0.90):] = 0   # tail
    torso[int(h * 0.70):, :] = 0   # legs
    torso[:int(h * 0.04), :] = 0   # top border

    out = Image.fromarray(torso).filter(ImageFilter.GaussianBlur(radius=8))
    out = Image.fromarray(((np.array(out) > 55) * 255).astype(np.uint8))
    return out.convert("L")


# ─── Step 3: ControlNet + SD Inpainting Pipeline ─────────────────────────────
_controlnet_pipe = None
def load_controlnet_pipeline():
    global _controlnet_pipe
    if _controlnet_pipe is not None:
        return _controlnet_pipe

    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler

    print("[ControlNet] Loading ControlNet depth model...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float32,
    )

    print("[ControlNet] Loading SD 1.5 Inpainting + ControlNet pipeline...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Faster scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing(slice_size=1)
    print("[ControlNet] Pipeline ready!\n")

    _controlnet_pipe = pipe
    return pipe


# ─── Step 4: Run VTON on One Image ───────────────────────────────────────────
def run_controlnet_vton(img_path: Path, out_name: str = None) -> Path:
    SD_SIZE = (512, 512)
    img_path = Path(img_path)
    out_name = out_name or f"{img_path.stem}_controlnet_vton.png"
    out_path = OUTPUT_DIR / out_name

    print(f"\n[VTON] Processing: {img_path.name}")

    # Load image
    dog_orig = Image.open(img_path).convert("RGB")
    orig_size = dog_orig.size
    dog_sd = dog_orig.resize(SD_SIZE, Image.LANCZOS)

    # Get depth map (this is what guides jacket draping)
    print("  → Estimating depth map...")
    depth_map = get_depth_map(dog_orig, SD_SIZE)
    depth_map.save(OUTPUT_DIR / f"{img_path.stem}_depth.png")
    print(f"  → Depth map saved")

    # Get torso mask
    print("  → Generating torso mask...")
    mask = get_torso_mask(img_path, SD_SIZE)
    mask.save(OUTPUT_DIR / f"{img_path.stem}_mask.png")

    # Load pipeline
    pipe = load_controlnet_pipeline()

    pos_prompt = (
        "photo of a dog wearing an olive green Ruffwear dog jacket coat vest, "
        "full back coverage from neck to tail base, jacket covers entire spine and torso, "
        "saddle-style dog coat, belly strap visible underneath, D-ring handle on back, "
        "olive army green waterproof fabric, black trim edges, velcro fasteners, "
        "jacket draped naturally over curved dog back, fur visible at collar and legs, "
        "realistic photographic quality, natural lighting, photorealistic 8k photo"
    )
    neg_prompt = (
        "cartoon, painting, illustration, anime, blurry, distorted, "
        "human clothes, deformed anatomy, watermark, text, low quality, ugly"
    )

    print("  → Running ControlNet inpainting...")
    t0 = time.time()

    with torch.no_grad():
        out = pipe(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            image=dog_sd,
            mask_image=mask,
            control_image=depth_map,      # ← depth guides jacket shape
            height=SD_SIZE[1],
            width=SD_SIZE[0],
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.45,  # lower = prompt dominates, depth guides shape
            strength=0.85,
            generator=torch.Generator(device=DEVICE).manual_seed(42),
        )

    gen_img = out.images[0]

    # Composite back to original resolution
    gen_fullres  = gen_img.resize(orig_size, Image.LANCZOS)
    mask_fullres = get_torso_mask(img_path, orig_size)
    mask_soft    = mask_fullres.filter(ImageFilter.GaussianBlur(radius=4))
    final = Image.composite(gen_fullres, dog_orig, mask_soft)
    final.save(out_path, format="PNG")

    elapsed = time.time() - t0
    print(f"  ✓ Saved: {out_path.name} — {elapsed:.1f}s")

    if DEVICE == "mps":
        torch.mps.empty_cache()

    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, help="Path to a single dog image")
    p.add_argument("--batch_golden", action="store_true", help="Run on all golden dataset images")
    p.add_argument("--batch_labrador", action="store_true", help="Run on all labrador dataset images")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.image:
        run_controlnet_vton(Path(args.image))

    elif args.batch_golden or args.batch_labrador:
        datasets = []
        if args.batch_golden:
            datasets.append(BASE_DIR / "CCBD_repo" / "golden")
        if args.batch_labrador:
            datasets.append(BASE_DIR / "CCBD_repo" / "labrador")

        for dataset_dir in datasets:
            imgs = sorted(list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.jpeg")))
            if args.limit:
                imgs = imgs[:args.limit]
            print(f"\n{'='*60}\n  DATASET: {dataset_dir.name} ({len(imgs)} images)\n{'='*60}")
            for i, img in enumerate(imgs, 1):
                print(f"\n[{i}/{len(imgs)}]")
                try:
                    run_controlnet_vton(img)
                except Exception as e:
                    print(f"  [ERROR] {img.name}: {e}")
    else:
        # Default: test on first golden image
        test_img = BASE_DIR / "CCBD_repo" / "golden" / "golden_golden_retriever_dog_show_bing_00002.jpg"
        run_controlnet_vton(test_img)
