import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import config
from utils import logger, save_state

# Thread-safe lock for state updates
state_lock = threading.Lock()

def fetch_image_urls(breed_api_path):
    """Fetch all image URLs for a given breed from Dog CEO API."""
    url = f"https://dog.ceo/api/breed/{breed_api_path}/images"
    try:
        response = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return data.get("message", [])
        else:
            logger.error(f"Failed to fetch breed list from Dog CEO: {data.get('message')}")
            return []
    except Exception as e:
        logger.error(f"Error communicating with Dog CEO API for {breed_api_path}: {e}")
        return []

def download_image(url, breed_folder, state, state_file):
    """Download a single image and update the state."""
    filename = url.split("/")[-1]
    save_path = os.path.join(config.RAW_DIR, breed_folder, filename)
    
    # Check if already downloaded and exists on disk
    with state_lock:
        if url in state["downloaded"] and os.path.exists(save_path):
            return "skipped"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        response = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        # Update state thread-safely
        with state_lock:
            state["downloaded"][url] = os.path.join("raw", breed_folder, filename)
            save_state(state, state_file)
            
        return "success"
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return "failed"

def download_all_images(state, state_file):
    """Orchestrates multi-threaded downloading for all configured breeds."""
    logger.info("Starting dog image download stage...")
    
    stats = {breed: {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0} for breed in config.BREEDS}
    
    for breed, breed_api_path in config.BREEDS.items():
        logger.info(f"Fetching URLs for {breed} ({breed_api_path})...")
        urls = fetch_image_urls(breed_api_path)
        stats[breed]["total"] = len(urls)
        
        if not urls:
            logger.warning(f"No URLs found or fetched for {breed}.")
            continue
            
        logger.info(f"Found {len(urls)} images. Starting download...")
        
        # Parallel downloads using thread pool
        with ThreadPoolExecutor(max_workers=config.DOWNLOAD_MAX_WORKERS) as executor:
            # Map future to url
            futures = {
                executor.submit(download_image, url, breed, state, state_file): url
                for url in urls
            }
            
            # Progress bar for this breed
            with tqdm(total=len(urls), desc=f"Downloading {breed}", unit="img") as pbar:
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        result = future.result()
                        if result == "success":
                            stats[breed]["downloaded"] += 1
                        elif result == "skipped":
                            stats[breed]["skipped"] += 1
                        else:
                            stats[breed]["failed"] += 1
                    except Exception as e:
                        logger.error(f"Exception during download task for {url}: {e}")
                        stats[breed]["failed"] += 1
                    
                    pbar.update(1)
                    
        logger.info(
            f"Finished downloading {breed}. "
            f"Downloaded: {stats[breed]['downloaded']}, "
            f"Skipped (Resume): {stats[breed]['skipped']}, "
            f"Failed: {stats[breed]['failed']}"
        )
        
    return stats
