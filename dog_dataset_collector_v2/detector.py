import os
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import config
from utils import logger, get_device, save_state

def detect_and_crop_dogs(state, state_file):
    """Processes downloaded raw images, detects dogs, crops and filters them using YOLOv8."""
    logger.info("Starting dog detection and cropping stage...")
    
    device = get_device()
    device_str = "cpu"
    if device.type == "mps":
        device_str = "mps"
    elif device.type == "cuda":
        device_str = "cuda"
        
    logger.info(f"Loading YOLO model: {config.YOLO_MODEL_NAME} on device: {device_str}")
    model = YOLO(config.YOLO_MODEL_NAME)
    
    # Track statistics
    stats = {
        "total_processed": 0,
        "skipped_resume": 0,
        "passed": 0,
        "rejected_no_dog": 0,
        "rejected_multi_dog": 0,
        "rejected_too_small": 0,
        "rejected_cut_off": 0,
        "errors": 0
    }
    
    # Iterate over breeds
    for breed in config.BREEDS.keys():
        raw_breed_dir = os.path.join(config.RAW_DIR, breed)
        cropped_breed_dir = os.path.join(config.CROPPED_DIR, breed)
        
        if not os.path.exists(raw_breed_dir):
            logger.warning(f"Raw directory does not exist for {breed}. Skipping.")
            continue
            
        os.makedirs(cropped_breed_dir, exist_ok=True)
        filenames = [f for f in os.listdir(raw_breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if not filenames:
            logger.warning(f"No raw images found for {breed}.")
            continue
            
        logger.info(f"Processing {len(filenames)} raw images for {breed}...")
        
        for filename in tqdm(filenames, desc=f"Detecting {breed}", unit="img"):
            raw_path = os.path.join(raw_breed_dir, filename)
            cropped_path = os.path.join(cropped_breed_dir, filename)
            
            # Check for resume support
            if filename in state["detected"]:
                saved_status = state["detected"][filename].get("status")
                # If it passed, verify the file exists on disk
                if saved_status == "passed" and os.path.exists(cropped_path):
                    stats["skipped_resume"] += 1
                    stats["passed"] += 1
                    continue
                elif saved_status != "passed":
                    stats["skipped_resume"] += 1
                    stats[f"rejected_{saved_status.split('rejected_')[-1]}"] += 1
                    continue
            
            stats["total_processed"] += 1
            
            try:
                # Open image to ensure it is valid
                with Image.open(raw_path) as img:
                    img = img.convert("RGB")
                    img_width, img_height = img.size
                    
                # Run YOLOv8 detection
                # Target class 16 (dog) in COCO dataset
                results = model.predict(
                    source=raw_path, 
                    device=device_str, 
                    conf=config.YOLO_CONF, 
                    classes=[16],  # COCO class 16 is dog
                    verbose=False
                )
                
                # Check detections
                if not results or len(results[0].boxes) == 0:
                    state["detected"][filename] = {"status": "rejected_no_dog"}
                    stats["rejected_no_dog"] += 1
                    save_state(state, state_file)
                    continue
                    
                dog_boxes = results[0].boxes
                
                if len(dog_boxes) > 1:
                    state["detected"][filename] = {"status": "rejected_multi_dog"}
                    stats["rejected_multi_dog"] += 1
                    save_state(state, state_file)
                    continue
                
                # Exactly one dog detected
                box = dog_boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 1. Area Check: must occupy >= 40% of the image
                box_area = (x2 - x1) * (y2 - y1)
                img_area = img_width * img_height
                area_ratio = box_area / img_area
                
                if area_ratio < config.YOLO_MIN_AREA_RATIO:
                    state["detected"][filename] = {
                        "status": "rejected_too_small",
                        "area_ratio": area_ratio
                    }
                    stats["rejected_too_small"] += 1
                    save_state(state, state_file)
                    continue
                
                # 2. Cut-off Check: bounding box must not touch image boundaries
                margin = config.YOLO_EDGE_MARGIN
                is_cut_off = (
                    x1 < margin or 
                    y1 < margin or 
                    x2 > img_width - margin or 
                    y2 > img_height - margin
                )
                
                if is_cut_off:
                    state["detected"][filename] = {
                        "status": "rejected_cut_off",
                        "box": [x1, y1, x2, y2],
                        "dims": [img_width, img_height]
                    }
                    stats["rejected_cut_off"] += 1
                    save_state(state, state_file)
                    continue
                
                # All checks passed, crop and save the dog
                with Image.open(raw_path) as img:
                    img = img.convert("RGB")
                    # Crop boundary box coordinates (x1, y1, x2, y2)
                    cropped_img = img.crop((x1, y1, x2, y2))
                    cropped_img.save(cropped_path, "JPEG", quality=95)
                    
                state["detected"][filename] = {
                    "status": "passed",
                    "crop_path": os.path.join("cropped", breed, filename),
                    "box": [x1, y1, x2, y2],
                    "area_ratio": area_ratio
                }
                stats["passed"] += 1
                save_state(state, state_file)
                
            except Exception as e:
                logger.error(f"Error processing image {filename} in YOLO detector: {e}")
                stats["errors"] += 1
                state["detected"][filename] = {"status": "rejected_error"}
                save_state(state, state_file)
                
    logger.info(
        f"YOLO Stage Finished. "
        f"Total: {stats['total_processed']}, "
        f"Passed: {stats['passed']}, "
        f"No Dog: {stats['rejected_no_dog']}, "
        f"Multi Dog: {stats['rejected_multi_dog']}, "
        f"Too Small: {stats['rejected_too_small']}, "
        f"Cut-off: {stats['rejected_cut_off']}, "
        f"Errors/Skipped: {stats['errors'] + stats['skipped_resume']}"
    )
    
    return stats
