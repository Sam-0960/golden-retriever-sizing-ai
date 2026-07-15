import os
import json
import logging
import torch

# Initialize Logger
def setup_logging(log_file="dataset_collector.log"):
    logger = logging.getLogger("dataset_collector")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file {log_file}: {e}")
        
    return logger

# Get Logger
logger = setup_logging()

# Device selection (MPS, CUDA, CPU)
def get_device():
    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU Acceleration (MPS).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA GPU Acceleration.")
        return torch.device("cuda")
    else:
        logger.info("GPU acceleration not detected. Using CPU.")
        return torch.device("cpu")

# Atomic JSON state functions
def load_state(state_file):
    if not os.path.exists(state_file):
        logger.info("No existing progress state found. Starting fresh.")
        return {
            "downloaded": {},   # url -> filename
            "detected": {},     # raw_filename -> { "status": "passed/rejected_...", "crop_path": "..." }
            "filtered": {},     # crop_filename -> { "status": "passed/rejected", "answer": "YES/NO" }
            "deduplicated": {}  # final_filename -> { "status": "passed/rejected_duplicate" }
        }
    
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
            logger.info("Successfully loaded progress state from disk.")
            # Ensure all keys exist
            for key in ["downloaded", "detected", "filtered", "deduplicated"]:
                if key not in state:
                    state[key] = {}
            return state
    except Exception as e:
        logger.error(f"Error loading state file: {e}. Starting fresh to be safe.")
        return {
            "downloaded": {},
            "detected": {},
            "filtered": {},
            "deduplicated": {}
        }

def save_state(state, state_file):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    temp_file = state_file + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, state_file)
    except Exception as e:
        logger.error(f"Failed to save state file atomically: {e}")
