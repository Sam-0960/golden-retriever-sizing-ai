import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Stage Directories
RAW_DIR = os.path.join(DATASET_DIR, "raw")
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped")
FINAL_DIR = os.path.join(DATASET_DIR, "final")

# Breed API mapping
# folder_name -> Dog CEO API path
BREEDS = {
    "golden_retriever": "retriever/golden",
    "labrador_retriever": "labrador"
}

# Resume State File
STATE_FILE = os.path.join(DATASET_DIR, "dataset_state.json")

# Downloader Settings
DOWNLOAD_MAX_WORKERS = 8
DOWNLOAD_TIMEOUT = 15

# Detector Settings (YOLOv8)
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_MIN_AREA_RATIO = 0.40
YOLO_EDGE_MARGIN = 5  # pixels

# Vision-Language Model Settings (Florence-2)
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE_PROMPT = (
    "Is this a full-body standing side-profile image of a Labrador Retriever or Golden Retriever "
    "suitable for single-image 3D reconstruction? Answer ONLY YES or NO."
)

# Deduplication Settings (CLIP)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_SIMILARITY_THRESHOLD = 0.95
DEDUPLICATE_BATCH_SIZE = 32
