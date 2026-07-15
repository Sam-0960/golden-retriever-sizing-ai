import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Intermediate and final paths
RAW_DIR = os.path.join(DATASET_DIR, "raw")
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped")
FILTERED_DIR = os.path.join(DATASET_DIR, "filtered")

# Search queries per breed
BREEDS = {
    "golden": [
        "golden retriever standing side profile",
        "golden retriever side view",
        "golden retriever standing",
        "golden retriever dog show"
    ],
    "labrador": [
        "labrador retriever standing side profile",
        "labrador side view",
        "labrador standing",
        "labrador dog show"
    ]
}

# Resume State File
STATE_FILE = os.path.join(DATASET_DIR, "dataset_state.json")

# Downloader Settings (icrawler)
MAX_IMAGES_PER_QUERY = 250  # 4 queries * 250 = ~1000 candidate images per breed
DOWNLOAD_THREADS = 8
DOWNLOAD_TIMEOUT = 10

# Detector Settings (YOLOv8)
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_MIN_AREA_RATIO = 0.40
YOLO_EDGE_MARGIN = 5  # pixels

# Vision-Language Model Settings (Florence-2)
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"

# Deduplication Settings (CLIP)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_SIMILARITY_THRESHOLD = 0.95
DEDUPLICATE_BATCH_SIZE = 32
