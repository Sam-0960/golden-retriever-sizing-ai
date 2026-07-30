#!/usr/bin/env python3
"""
Quick screening script: picks best side-profile dog images from labrador dataset.
Criteria:
  - Dog detected by YOLO (class 16)
  - Bounding box is wider than tall (= side view, not face-on or angled)
  - Dog occupies at least 20% of frame (not too far away)
  - Box aspect ratio >= 1.3 (clearly horizontal = good side profile)
"""

from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import json

LABRADOR_DIR = Path("/Users/udayk/Desktop/CCBD/CCBD_repo/labrador")
model = YOLO("/Users/udayk/Desktop/CCBD/yolov8n-seg.pt")

imgs = sorted(list(LABRADOR_DIR.glob("*.jpg")) + list(LABRADOR_DIR.glob("*.jpeg")))
print(f"Total labrador images: {len(imgs)}")

scored = []
for img_path in imgs:
    try:
        img = Image.open(img_path)
        W, H = img.size
        frame_area = W * H

        results = model(img_path, verbose=False)
        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        # find dog box
        dog_box = None
        for i, cls_id in enumerate(boxes.cls):
            if int(cls_id) == 16:
                dog_box = boxes.xyxy[i].tolist()
                break

        if dog_box is None:
            # fallback: take the largest box
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy.tolist()]
            dog_box = boxes.xyxy[areas.index(max(areas))].tolist()

        x1, y1, x2, y2 = dog_box
        bw = x2 - x1
        bh = y2 - y1
        aspect = bw / (bh + 1e-6)
        coverage = (bw * bh) / frame_area

        score = aspect * coverage  # high score = wide dog, large in frame
        scored.append({
            "file": img_path.name,
            "path": str(img_path),
            "aspect": round(aspect, 2),
            "coverage": round(coverage, 2),
            "score": round(score, 3),
            "size": f"{W}x{H}"
        })
        print(f"  {img_path.name}: aspect={aspect:.2f}, coverage={coverage:.1%}, score={score:.3f}")

    except Exception as e:
        print(f"  [ERROR] {img_path.name}: {e}")

# Sort by score descending
scored.sort(key=lambda x: x["score"], reverse=True)

print(f"\n{'='*60}")
print(f"TOP 20 BEST SIDE-PROFILE IMAGES:")
print(f"{'='*60}")
top20 = scored[:20]
for i, s in enumerate(top20, 1):
    print(f"  {i:2}. {s['file']} — aspect={s['aspect']}, coverage={s['coverage']:.1%}, score={s['score']}")

# Save list
out = Path("/Users/udayk/Desktop/CCBD/labrador_top20.json")
with open(out, "w") as f:
    json.dump(top20, f, indent=2)
print(f"\nSaved: {out}")
