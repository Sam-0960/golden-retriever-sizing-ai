# Dog Dataset Collector

An automated, end-to-end Python pipeline that scrapes Google and Bing for high-quality, full-body standing side-profile images of Golden Retrievers and Labrador Retrievers. The collected dataset is optimized for single-image 3D reconstruction models such as [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

The pipeline is fully resumable and runs end-to-end: **Scrape → Detect → Crop → Pose Filter → Deduplicate → (optional) 3D Reconstruction**.

---

## Features

- **Auto-Dependency Installation** — running `python run_dataset.py` checks for missing modules, installs them, and resumes execution.
- **Multi-Engine Scraping** — Google Images and Bing Images are scraped in parallel via [`icrawler`](https://github.com/hkchengrex/Crawler).
- **YOLOv8 Single-Dog Detection & Crop** — keeps images with exactly one dog that fills at least 40% of the frame and is not cut off at any edge.
- **Florence-2 VLM Pose Filter** — generates a detailed visual caption and rejects sit, lying, or close-up portraits, keeping only standing side-profile views.
- **CLIP-Based Deduplication** — drops near-duplicate images using cosine similarity (`> 0.95`).
- **Apple Silicon (MPS) / CUDA / CPU Acceleration** — auto-selects the best available device with a graceful CPU fallback.
- **Resumable Pipeline** — progress is checkpointed to `dataset/dataset_state.json` so interrupted runs pick up where they left off.
- **Optional TripoSR 3D Reconstruction** — `run_triposr_on_dataset.py` turns the first 10 images per breed into `.glb` meshes.

---

## Pipeline Overview

```
Google + Bing (icrawler)
        │
        ▼
  dataset/raw/<breed>/              ← raw scraped images
        │
        ▼
  YOLOv8 detect + crop
        │
        ▼
  dataset/cropped/<breed>/          ← single-dog crops
        │
        ▼
  Florence-2 pose filter
        │
        ▼
  dataset/filtered/<breed>/         ← standing side-profile only
        │
        ▼
  CLIP deduplication
        │
        ▼
  dataset/<breed>/00001.jpg …       ← final clean dataset
        │
        ▼
  TripoSR 3D reconstruction (optional)
        │
        ▼
  dataset_3d/<breed>_xxxxx_mesh.glb
```

---

## Folder Structure

After a complete run, your working directory will look like:

```
dog_dataset_collector/
├── README.md
├── config.py
├── downloader.py
├── detector.py
├── filter.py
├── deduplicate.py
├── run_dataset.py
├── run_triposr_on_dataset.py
├── utils.py
├── requirements.txt
├── yolov8n.pt
├── dataset/                      ← gitignored (regenerated)
│   ├── golden/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── …
│   ├── labrador/
│   │   ├── 00001.jpg
│   │   └── …
│   ├── raw/                      ← intermediate
│   ├── cropped/                  ← intermediate
│   ├── filtered/                 ← intermediate
│   └── dataset_state.json        ← resume checkpoint
└── dataset_3d/                   ← gitignored (3D outputs)
    ├── golden_00001_mesh.glb
    └── …
```

> `dataset/` and `dataset_3d/` are listed in `.gitignore` and are never pushed to the repository. The pipeline regenerates them on each run.

---

## Requirements

- **Python 3.11+**
- A working internet connection (first run downloads model weights)
- Optional: Apple Silicon Mac for MPS acceleration, or an NVIDIA GPU for CUDA

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/dog-dataset-collector.git
cd dog-dataset-collector
```

### 2. Run the dataset collector

```bash
python run_dataset.py
```

On the first run the script will:

1. Install any missing dependencies (`icrawler`, `transformers`, `ultralytics`, `torch`, `timm`, `einops`, …).
2. Download the YOLOv8, Florence-2, and CLIP model files into the local Hugging Face / Ultralytics cache.
3. Execute the full pipeline.

Models are cached locally, so subsequent runs skip the downloads.

To re-run only a subset of stages, simply delete the relevant subfolders under `dataset/` (and `dataset/dataset_state.json` for a clean slate) and re-run.

---

## Configuration

All tunables live in [`config.py`](config.py). Notable knobs:

| Setting                       | Default                                | Description                                                       |
| ----------------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| `BREEDS`                      | `golden`, `labrador`                   | Search queries per breed.                                         |
| `MAX_IMAGES_PER_QUERY`        | `250`                                  | Candidate images per query (× 2 engines × 4 queries ≈ 2 000/breed). |
| `YOLO_CONF`                   | `0.25`                                 | YOLOv8 confidence threshold.                                      |
| `YOLO_MIN_AREA_RATIO`         | `0.40`                                 | Minimum fraction of the image the dog must occupy.                |
| `YOLO_EDGE_MARGIN`            | `5` px                                 | Bounding box must stay this many pixels from any image edge.      |
| `FLORENCE_MODEL_ID`           | `microsoft/Florence-2-base`            | Vision-language model used for pose filtering.                    |
| `CLIP_MODEL_ID`               | `openai/clip-vit-base-patch32`         | CLIP backbone used for deduplication.                             |
| `CLIP_SIMILARITY_THRESHOLD`   | `0.95`                                 | Cosine similarity above which images are treated as duplicates.   |

---

## Optional: 3D Reconstruction with TripoSR

After `dataset/<breed>/` is populated, you can generate 3D meshes for the first 10 images of each breed using [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

1. Clone and set up TripoSR locally (see its README).
2. Point the script to your installation:

   ```bash
   export TRIPOSR_DIR="$HOME/path/to/TripoSR"
   ```

3. Run the reconstruction:

   ```bash
   python run_triposr_on_dataset.py
   ```

   The script uses TripoSR's own virtual-environment Python to load `rembg` and TripoSR, then saves `.glb` files into `dataset_3d/`.

---

## Resuming an Interrupted Run

State is checkpointed atomically to `dataset/dataset_state.json`. Just re-run:

```bash
python run_dataset.py
```

…and the pipeline will skip any images that have already been downloaded, cropped, filtered, or deduplicated.

---

## Output Summary

At the end of every run, the pipeline prints a summary like:

```
================================================================================
                     DATASET COLLECTION RUN SUMMARY REPORT (v2)
================================================================================
Images Downloaded:                    1842
Images Rejected:                      1573
Duplicate Images Removed:             42
Images Accepted:                      227

Rejections Breakdown:
  - YOLOv8 Rejections:                1502
      * No dog detected:              712
      * Multiple dogs detected:       188
      * Dog too small (<40% area):    402
      * Dog cut off at borders:       200
  - Florence-2 VLM Rejections:        71
      * Pose/Angle mismatch:          71
================================================================================
Final Counts by Breed:
  - Golden:   112 images
  - Labrador: 115 images
================================================================================
```

---

## License

Released under the [MIT License](LICENSE).
