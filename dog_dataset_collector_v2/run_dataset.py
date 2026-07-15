import os
import sys
import subprocess

# 1. Dependency Auto-Installer (runs first using only standard python library)
def check_and_install_dependencies():
    required_packages = {
        "icrawler": "icrawler",
        "ultralytics": "ultralytics",
        "transformers": "transformers==4.45.2",
        "tqdm": "tqdm",
        "requests": "requests",
        "PIL": "pillow",
        "einops": "einops",
        "timm": "timm"
    }
    
    missing = []
    for module_name, pip_name in required_packages.items():
        # pillow is imported as PIL
        check_name = "PIL" if module_name == "PIL" else module_name
        try:
            __import__(check_name)
        except ImportError:
            missing.append(pip_name)
            
    if missing:
        print("==================================================")
        print("Missing dependencies detected! Installing them now...")
        print(f"Installing: {', '.join(missing)}")
        print("==================================================")
        try:
            # Execute pip install
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("\nDependencies installed successfully! Restarting script...")
            print("==================================================\n")
            # Restart current script
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"Error during automatic dependency installation: {e}")
            print("Please run manually: pip install -r requirements.txt")
            sys.exit(1)

# Run dependency check
check_and_install_dependencies()

# 2. Imports after dependency checks are complete
import config
from utils import logger, load_state, save_state
from downloader import download_all_images
from detector import detect_and_crop_dogs
from filter import run_side_view_filtering
from deduplicate import remove_duplicates_and_save

def main():
    logger.info("==================================================")
    logger.info("  Starting Dog Dataset Collector v2 (icrawler)    ")
    logger.info("==================================================")
    
    # Initialize stages folders
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.CROPPED_DIR, exist_ok=True)
    os.makedirs(config.FILTERED_DIR, exist_ok=True)
    
    state = load_state(config.STATE_FILE)
    
    try:
        # Step 1: Download candidates via icrawler (Google/Bing)
        download_all_images(state, config.STATE_FILE)
        
        # Step 2: YOLOv8 dog detection and cropping
        detect_and_crop_dogs(state, config.STATE_FILE)
        
        # Step 3: Florence-2 pose filtering
        run_side_view_filtering(state, config.STATE_FILE)
        
        # Step 4: CLIP duplicate removal
        remove_duplicates_and_save(state, config.STATE_FILE)
        
        # Step 5: Report final pipeline statistics
        print_final_report(state)
        
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user. Progress saved. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline crashed: {e}", exc_info=True)
        sys.exit(1)

def print_final_report(state):
    """Tally and display run statistics from state JSON."""
    # Count total images downloaded
    # downloaded state structures: {"golden": {"status": "completed", "raw_count": 123}, "labrador": ...}
    total_downloaded = sum(info.get("raw_count", 0) for info in state.get("downloaded", {}).values() if isinstance(info, dict))
    
    # YOLO Tally
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
            
    # Florence-2 Tally
    florence_stats = {"passed": 0, "rejected": 0, "error": 0}
    for file, info in state.get("filtered", {}).items():
        status = info.get("status", "")
        if status == "passed":
            florence_stats["passed"] += 1
        elif status == "rejected":
            florence_stats["rejected"] += 1
        else:
            florence_stats["error"] += 1
            
    # CLIP Deduplication Tally
    dedup_stats = {"passed": 0, "duplicate": 0}
    for file, info in state.get("deduplicated", {}).items():
        status = info.get("status", "")
        if status == "passed":
            dedup_stats["passed"] += 1
        elif status == "rejected_duplicate":
            dedup_stats["duplicate"] += 1
            
    # Final saved files count
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
                     DATASET COLLECTION RUN SUMMARY REPORT (v2)
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
      * Pose/Angle mismatch:          {florence_stats['rejected']}
      * VLM query errors:             {florence_stats['error']}

CLIP Deduplication:
  - Similarity rejections (>0.95):   {dedup_stats['duplicate']}

================================================================================
Final Counts by Breed:
"""
    for breed, count in final_counts.items():
        report += f"  - {breed.title()}: {count} images\n"
    report += "================================================================================"
    
    print(report)

if __name__ == "__main__":
    main()
