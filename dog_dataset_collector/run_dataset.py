import os
import sys
import config
from utils import logger, load_state, save_state
from downloader import download_all_images
from detector import detect_and_crop_dogs
from filter import run_side_view_filtering
from deduplicate import remove_duplicates_and_save

def main():
    logger.info("==================================================")
    logger.info("  Starting Automated Dog Dataset Collector        ")
    logger.info("==================================================")
    
    # Initialize all required directories
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.CROPPED_DIR, exist_ok=True)
    os.makedirs(config.FINAL_DIR, exist_ok=True)
    
    # Load resume state
    state = load_state(config.STATE_FILE)
    
    try:
        # Step 1: Download images from Dog CEO API
        download_all_images(state, config.STATE_FILE)
        
        # Step 2: Detect & Crop dogs using YOLOv8
        detect_and_crop_dogs(state, config.STATE_FILE)
        
        # Step 3: Filter side profiles using Florence-2 VLM
        run_side_view_filtering(state, config.STATE_FILE)
        
        # Step 4: Remove duplicates using CLIP embeddings
        remove_duplicates_and_save(state, config.STATE_FILE)
        
        # Step 5: Generate and print summary report
        print_final_report(state)
        
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user. Progress has been saved. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline crashed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)

def print_final_report(state):
    """Parses current state records and final output folders to print a clean run report."""
    total_downloaded = len(state.get("downloaded", {}))
    
    # Tally YOLO rejections
    yolo_stats = {"no_dog": 0, "multi_dog": 0, "too_small": 0, "cut_off": 0, "passed": 0, "error": 0}
    for file, info in state.get("detected", {}).items():
        status = info.get("status", "")
        if status == "passed":
            yolo_stats["passed"] += 1
        elif "rejected_" in status:
            reason = status.replace("rejected_", "")
            yolo_stats[reason] = yolo_stats.get(reason, 0) + 1
        elif status in ("rejected_error", "error"):
            yolo_stats["error"] += 1
            
    # Tally Florence-2 VLM rejections
    florence_stats = {"passed": 0, "rejected": 0, "error": 0}
    for file, info in state.get("filtered", {}).items():
        status = info.get("status", "")
        if status == "passed":
            florence_stats["passed"] += 1
        elif status == "rejected":
            florence_stats["rejected"] += 1
        else:
            florence_stats["error"] += 1
            
    # Tally CLIP duplicates
    dedup_stats = {"passed": 0, "duplicate": 0}
    for file, info in state.get("deduplicated", {}).items():
        status = info.get("status", "")
        if status == "passed":
            dedup_stats["passed"] += 1
        elif status == "rejected_duplicate":
            dedup_stats["duplicate"] += 1
            
    # Tally final breed folder files
    final_counts = {}
    for breed in config.BREEDS.keys():
        breed_dir = os.path.join(config.DATASET_DIR, breed)
        if os.path.exists(breed_dir):
            final_counts[breed] = len([f for f in os.listdir(breed_dir) if f.lower().endswith('.jpg')])
        else:
            final_counts[breed] = 0
            
    total_rejected = (
        (yolo_stats["no_dog"] + yolo_stats["multi_dog"] + yolo_stats["too_small"] + yolo_stats["cut_off"] + yolo_stats["error"]) +
        (florence_stats["rejected"] + florence_stats["error"])
    )
    
    report = f"""
================================================================================
                     DATASET COLLECTION RUN SUMMARY REPORT
================================================================================
Images Downloaded:                    {total_downloaded}
Images Rejected:                      {total_rejected}
Duplicate Images Removed:             {dedup_stats['duplicate']}
Images Accepted:                      {sum(final_counts.values())}

Rejections Breakdown:
  - YOLOv8 Rejections:                {yolo_stats['no_dog'] + yolo_stats['multi_dog'] + yolo_stats['too_small'] + yolo_stats['cut_off'] + yolo_stats['error']}
      * No dog detected:              {yolo_stats['no_dog']}
      * Multiple dogs detected:       {yolo_stats['multi_dog']}
      * Dog too small (<40% area):    {yolo_stats['too_small']}
      * Dog cut off at borders:       {yolo_stats['cut_off']}
      * Processing errors:            {yolo_stats['error']}
  - Florence-2 VLM Rejections:        {florence_stats['rejected'] + florence_stats['error']}
      * Not a side profile / pose:    {florence_stats['rejected']}
      * VLM query errors:             {florence_stats['error']}

CLIP Deduplication:
  - Similarity rejections:            {dedup_stats['duplicate']}

================================================================================
Final Counts by Breed:
"""
    for breed, count in final_counts.items():
        report += f"  - {breed.replace('_', ' ').title()}: {count} images\n"
    report += "================================================================================"
    
    print(report)

if __name__ == "__main__":
    main()
