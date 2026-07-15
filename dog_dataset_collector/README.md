# Dog Dataset Collector for TripoSR Reconstruction

An automated, end-to-end pipeline to build a clean, high-quality dataset of side-profile standing images of Golden Retrievers and Labrador Retrievers, optimized for single-image 3D reconstruction (e.g., TripoSR).

It automatically downloads images from the Dog CEO API, detects and crops single dogs using YOLOv8, filters side-view poses using the Florence-2 vision-language model, removes near-identical duplicates using CLIP, and organizes the final dataset.

## Final Folder Structure
Once the pipeline runs, the final dataset is structured as:
```
dataset/
├── golden_retriever/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
└── labrador_retriever/
    ├── 00001.jpg
    ├── 00002.jpg
    └── ...
```

---

## Features

1. **Dog CEO API Fetching**: Automatically fetches all available retriever-golden and labrador images without web scraping.
2. **YOLOv8 Dog Detection**:
   - Detects dogs in the image.
   - Enforces exactly **one** dog per image.
   - Rejects images where the dog is too small (occupying < 40% of image area).
   - Rejects images where the dog's body is cut off (bounding box touches image boundaries).
   - Crops tightly around the dog.
3. **Florence-2 VLM Filtering**:
   - Queries `microsoft/Florence-2-base` locally to ask whether the cropped image is a full-body standing side-profile view of the target breeds suitable for 3D reconstruction.
   - Automatically filters out front views, rear views, close-ups, sitting/lying dogs, blurry shots, and heavily occluded views.
4. **CLIP Deduplication**:
   - Computes normalized embeddings using `openai/clip-vit-base-patch32`.
   - Discards images with a cosine similarity > 0.95.
5. **Apple Silicon Optimization**:
   - Detects and runs PyTorch operations on `mps` (Metal Performance Shaders) for fast execution on Apple Silicon (M1/M2/M3/M4).
   - Implements robust error catching; if MPS encounters unsupported model operators in Florence-2 or CLIP, it falls back to CPU automatically without crashing.
6. **Resume Support**:
   - Tracks pipeline progress in `dataset/dataset_state.json`.
   - If interrupted, restarting the script will skip already processed stages and completed images.

---

## Installation & Setup

We recommend using the ultra-fast Python package installer `uv`, which is already available on your machine.

1. **Activate the Virtual Environment**:
   ```bash
   cd path/to/dog_dataset_collector
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
   *Note: This will install torch, torchvision, transformers, ultralytics, and all required model drivers.*

---

## How to Run

Simply execute the main orchestration script:

```bash
python run_dataset.py
```

### Flow Breakdown:
1. **Downloader**: Calls Dog CEO API and downloads images concurrently using a thread pool.
2. **Detector**: Loads YOLOv8, detects single dogs, validates dimensions, crops, and saves them.
3. **VLM Filter**: Loads Florence-2, evaluates side-view poses, and saves candidates.
4. **Deduplicator**: Loads CLIP, runs a similarity grid check, drops matches > 0.95, and outputs numbered JPGs (`00001.jpg`, etc.).
5. **Summary Report**: Prints detailed download, rejection, and final counts.

---

## Customizing Thresholds

You can edit `config.py` to change parameters:
- `YOLO_MIN_AREA_RATIO` (default `0.40`): Enforces minimum size of the dog in the image.
- `YOLO_EDGE_MARGIN` (default `5`): Border proximity threshold to prevent cropping cut-off bodies.
- `CLIP_SIMILARITY_THRESHOLD` (default `0.95`): Similarity cutoff for duplicate detection.
