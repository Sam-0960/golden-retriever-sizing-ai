import os
import sys
import torch

def test_imports():
    print("Testing imports...")
    try:
        import config
        import utils
        import downloader
        import detector
        import filter
        import deduplicate
        print("  All project modules imported successfully!")
    except Exception as e:
        print(f"  Error importing modules: {e}")
        sys.exit(1)

def test_device():
    print("Checking device configuration...")
    from utils import get_device
    device = get_device()
    print(f"  Detected device: {device}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

def test_mock_pipeline():
    print("Running pipeline dry-run...")
    import config
    from utils import load_state
    
    # Temporarily modify configuration for a fast test
    config.DOWNLOAD_MAX_WORKERS = 2
    
    # 1. Fetch URLs for Golden Retriever and Labrador Retriever (just first 2)
    from downloader import fetch_image_urls, download_image
    
    golden_urls = fetch_image_urls(config.BREEDS["golden_retriever"])
    labrador_urls = fetch_image_urls(config.BREEDS["labrador_retriever"])
    
    print(f"  Fetched {len(golden_urls)} Golden Retriever URLs and {len(labrador_urls)} Labrador URLs.")
    
    if not golden_urls or not labrador_urls:
        print("  Error: Could not retrieve breed URLs from API.")
        sys.exit(1)
        
    # We will download just 1 image of each for our dry-run
    test_urls = {
        "golden_retriever": golden_urls[0],
        "labrador_retriever": labrador_urls[0]
    }
    
    state = load_state(config.STATE_FILE)
    
    print("  Downloading test images...")
    for breed, url in test_urls.items():
        res = download_image(url, breed, state, config.STATE_FILE)
        print(f"    Downloaded {breed} URL: {res}")
        
    # 2. Test YOLOv8 Detector
    print("  Running YOLOv8 dog detection...")
    from detector import detect_and_crop_dogs
    detector_stats = detect_and_crop_dogs(state, config.STATE_FILE)
    print(f"    YOLO stats: {detector_stats}")
    
    # 3. Test Florence-2 VLM Filter
    print("  Running Florence-2 VLM side-view pose filtering...")
    from filter import run_side_view_filtering
    filter_stats = run_side_view_filtering(state, config.STATE_FILE)
    print(f"    Florence-2 stats: {filter_stats}")
    
    # 4. Test CLIP Deduplication
    print("  Running CLIP duplicate detection...")
    from deduplicate import remove_duplicates_and_save
    dedup_stats = remove_duplicates_and_save(state, config.STATE_FILE)
    print(f"    CLIP stats: {dedup_stats}")
    
    # 5. Print final report
    print("  Generating summary report:")
    from run_dataset import print_final_report
    print_final_report(state)

if __name__ == "__main__":
    test_imports()
    test_device()
    test_mock_pipeline()
