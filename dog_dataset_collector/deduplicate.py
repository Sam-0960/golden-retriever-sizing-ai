import os
import shutil
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import config
from utils import logger, get_device, save_state

class CLIPDeduplicator:
    def __init__(self):
        self.device = get_device()
        self.model_id = config.CLIP_MODEL_ID
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Loads CLIP model with fallback to CPU on MPS failure."""
        logger.info(f"Loading CLIP model ({self.model_id}) on {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            logger.info("CLIP loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load CLIP on {self.device} due to: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def get_embeddings(self, image_paths):
        """Extracts and normalizes CLIP embeddings for a list of image paths in batches."""
        embeddings = []
        valid_paths = []
        
        # Batch process images
        for i in range(0, len(image_paths), config.DEDUPLICATE_BATCH_SIZE):
            batch_paths = image_paths[i:i + config.DEDUPLICATE_BATCH_SIZE]
            batch_images = []
            
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(p)
                except Exception as e:
                    logger.error(f"Error loading image {p} for CLIP embedding: {e}")
                    
            if not batch_images:
                continue
                
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                # Move input tensors to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Extract image features
                    features = self.model.get_image_features(**inputs)
                    # Normalize embeddings
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(features.cpu())
            except Exception as e:
                # Fallback to CPU if MPS fails during inference
                if self.device.type != "cpu":
                    logger.warning(f"CLIP inference failed on {self.device}: {e}. Retrying batch on CPU...")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
                    
                    # Try batch again on CPU
                    try:
                        inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            features = self.model.get_image_features(**inputs)
                            features = features / features.norm(p=2, dim=-1, keepdim=True)
                            embeddings.append(features.cpu())
                    except Exception as ex:
                        logger.error(f"Failed CLIP inference on CPU fallback: {ex}")
                else:
                    logger.error(f"CLIP inference failed: {e}")
                    
        if not embeddings:
            return None, []
            
        return torch.cat(embeddings, dim=0), valid_paths

def remove_duplicates_and_save(state, state_file):
    """Computes CLIP similarities, drops duplicates, and saves final dataset with sequential names."""
    logger.info("Starting duplicate removal stage (CLIP)...")
    
    filtered_dir = os.path.join(config.DATASET_DIR, "filtered")
    
    # Initialize the CLIP model
    deduplicator = CLIPDeduplicator()
    
    stats = {
        "total_filtered": 0,
        "duplicates_removed": 0,
        "final_saved": {},
        "skipped_resume": 0
    }
    
    for breed in config.BREEDS.keys():
        filtered_breed_dir = os.path.join(filtered_dir, breed)
        final_breed_dir = os.path.join(config.DATASET_DIR, breed)  # dataset/golden_retriever/ etc.
        
        # Setup output directory
        os.makedirs(final_breed_dir, exist_ok=True)
        
        if not os.path.exists(filtered_breed_dir):
            logger.warning(f"Filtered directory does not exist for {breed}. Skipping.")
            continue
            
        # Grab all files in the filtered directory
        filenames = sorted([f for f in os.listdir(filtered_breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not filenames:
            logger.warning(f"No filtered images found for {breed}.")
            continue
            
        stats["total_filtered"] += len(filenames)
        logger.info(f"Processing duplicate detection for {len(filenames)} images of {breed}...")
        
        # Check if already processed and matches exact final output files
        final_files_present = sorted([f for f in os.listdir(final_breed_dir) if f.lower().endswith('.jpg')])
        
        # Resume optimization: check if all filenames have state records in deduplicated
        already_deduped = True
        for f in filenames:
            if f not in state["deduplicated"]:
                already_deduped = False
                break
                
        if already_deduped and len(final_files_present) > 0:
            logger.info(f"Deduplication for {breed} already complete (Resume). Skipping.")
            
            # Count duplicates based on state records
            dup_count = sum(1 for f in filenames if state["deduplicated"].get(f, {}).get("status") == "rejected_duplicate")
            stats["duplicates_removed"] += dup_count
            stats["final_saved"][breed] = len(final_files_present)
            stats["skipped_resume"] += len(filenames)
            continue
            
        # Get absolute paths of images to compute CLIP embeddings
        image_paths = [os.path.join(filtered_breed_dir, f) for f in filenames]
        
        # Extract embeddings
        logger.info(f"Extracting CLIP embeddings for {breed}...")
        embeddings, valid_paths = deduplicator.get_embeddings(image_paths)
        
        if embeddings is None or len(valid_paths) == 0:
            logger.warning(f"No valid embeddings extracted for {breed}.")
            continue
            
        # Map valid paths back to base filenames
        valid_filenames = [os.path.basename(p) for p in valid_paths]
        num_images = len(valid_filenames)
        
        # Compute cosine similarity matrix
        # Since embeddings are L2 normalized, similarity is the dot product
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        keep_indices = []
        removed_indices = set()
        
        # Greedy deduplication
        for i in range(num_images):
            if i in removed_indices:
                continue
            keep_indices.append(i)
            for j in range(i + 1, num_images):
                if j in removed_indices:
                    continue
                similarity = similarity_matrix[i, j].item()
                if similarity > config.CLIP_SIMILARITY_THRESHOLD:
                    removed_indices.add(j)
                    # Mark duplicate in state
                    state["deduplicated"][valid_filenames[j]] = {
                        "status": "rejected_duplicate",
                        "similarity_score": similarity,
                        "duplicate_of": valid_filenames[i]
                    }
                    
        # Update statistics
        stats["duplicates_removed"] += len(removed_indices)
        
        # Clear out the final breed directory first to ensure clean numbering
        # only if we are rewriting it
        for f in os.listdir(final_breed_dir):
            if f.lower().endswith('.jpg'):
                try:
                    os.remove(os.path.join(final_breed_dir, f))
                except Exception as e:
                    logger.warning(f"Could not remove old file {f}: {e}")
                    
        # Save unique images with sequential names: 00001.jpg, 00002.jpg...
        logger.info(f"Saving unique images for {breed}...")
        saved_count = 0
        for idx in keep_indices:
            orig_filename = valid_filenames[idx]
            src_path = os.path.join(filtered_breed_dir, orig_filename)
            
            saved_count += 1
            final_filename = f"{saved_count:05d}.jpg"
            dest_path = os.path.join(final_breed_dir, final_filename)
            
            try:
                shutil.copy2(src_path, dest_path)
                state["deduplicated"][orig_filename] = {
                    "status": "passed",
                    "final_filename": final_filename,
                    "final_path": os.path.join(breed, final_filename)
                }
            except Exception as e:
                logger.error(f"Failed to copy final image {orig_filename} to {final_filename}: {e}")
                saved_count -= 1
                
        stats["final_saved"][breed] = saved_count
        save_state(state, state_file)
        
        logger.info(f"Breed {breed} complete. Saved {saved_count} unique images, removed {len(removed_indices)} duplicates.")
        
    return stats
