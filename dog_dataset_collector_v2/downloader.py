import os
import shutil
import logging
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import config
from utils import logger, save_state

# Suppress noisy icrawler logging
logging.getLogger('icrawler').setLevel(logging.WARNING)

def slugify(text):
    """Converts search queries to a clean filesystem-friendly string."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")

def download_breed_images(breed, queries, state, state_file):
    """Downloads candidate images for a single breed using Google and Bing."""
    if state["downloaded"].get(breed):
        logger.info(f"Downloads for breed '{breed}' already marked as completed (Resume). Skipping.")
        return
        
    logger.info(f"Starting downloads for breed '{breed}' using icrawler...")
    
    breed_raw_dir = os.path.join(config.RAW_DIR, breed)
    temp_dir = os.path.join(config.DATASET_DIR, "temp", breed)
    
    os.makedirs(breed_raw_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download queries
    # For 4 queries, we download up to config.MAX_IMAGES_PER_QUERY / 2 per engine
    max_per_engine = max(10, config.MAX_IMAGES_PER_QUERY // 2)
    
    for query in queries:
        query_slug = slugify(query)
        logger.info(f"Scraping query: '{query}' (up to {max_per_engine} images per engine)")
        
        # Google Crawl
        google_dest = os.path.join(temp_dir, f"{query_slug}_google")
        os.makedirs(google_dest, exist_ok=True)
        try:
            google_crawler = GoogleImageCrawler(
                feeder_threads=1,
                parser_threads=2,
                downloader_threads=config.DOWNLOAD_THREADS,
                storage={'root_dir': google_dest}
            )
            google_crawler.crawl(keyword=query, max_num=max_per_engine)
        except Exception as e:
            logger.error(f"Google Image search failed for '{query}': {e}")
            
        # Bing Crawl
        bing_dest = os.path.join(temp_dir, f"{query_slug}_bing")
        os.makedirs(bing_dest, exist_ok=True)
        try:
            bing_crawler = BingImageCrawler(
                feeder_threads=1,
                parser_threads=2,
                downloader_threads=config.DOWNLOAD_THREADS,
                storage={'root_dir': bing_dest}
            )
            bing_crawler.crawl(keyword=query, max_num=max_per_engine)
        except Exception as e:
            logger.error(f"Bing Image search failed for '{query}': {e}")

    # Aggregate all downloaded images into main raw directory
    logger.info(f"Aggregating and renaming downloaded candidates for '{breed}'...")
    total_aggregated = 0
    
    for subfolder in os.listdir(temp_dir):
        subfolder_path = os.path.join(temp_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
            
        # Iterate over downloaded files in query folder
        files = sorted(os.listdir(subfolder_path))
        for idx, filename in enumerate(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
                
            unique_name = f"{breed}_{subfolder}_{idx:05d}{ext}"
            src_path = os.path.join(subfolder_path, filename)
            dest_path = os.path.join(breed_raw_dir, unique_name)
            
            try:
                # Copy/move to main raw folder
                shutil.move(src_path, dest_path)
                total_aggregated += 1
            except Exception as e:
                logger.error(f"Failed to move image {filename} from {subfolder}: {e}")
                
    # Clean up temp folder
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Could not delete temp directory {temp_dir}: {e}")
        
    state["downloaded"][breed] = {
        "status": "completed",
        "raw_count": total_aggregated
    }
    save_state(state, state_file)
    logger.info(f"Finished downloads for '{breed}'. Aggregated {total_aggregated} images.")

def download_all_images(state, state_file):
    """Downloads candidates for all breeds."""
    # Ensure raw directory parent exists
    os.makedirs(config.RAW_DIR, exist_ok=True)
    
    for breed, queries in config.BREEDS.items():
        download_breed_images(breed, queries, state, state_file)
    
    # Return count statistics
    stats = {}
    for breed in config.BREEDS.keys():
        stats[breed] = state["downloaded"].get(breed, {}).get("raw_count", 0)
    return stats
