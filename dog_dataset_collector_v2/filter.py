import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import config
from utils import logger, get_device, save_state

class FlorenceFilter:
    def __init__(self):
        self.device = get_device()
        self.model_id = config.FLORENCE_MODEL_ID
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Loads Florence-2 model with fallback to CPU on MPS failure."""
        logger.info(f"Loading Florence-2 model ({self.model_id}) on {self.device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            logger.info("Florence-2 loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Florence-2 on {self.device} due to: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )

    def generate_caption(self, image):
        """Generates a detailed caption for the image with automatic device fallback."""
        try:
            return self._run_inference(image)
        except Exception as e:
            if self.device.type != "cpu":
                logger.warning(f"Florence-2 inference failed on {self.device} with error: {e}. Re-trying on CPU...")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                return self._run_inference(image)
            else:
                raise e

    def _run_inference(self, image):
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        
        # Move tensors to the correct device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Forward pass / Generation
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=150,
            num_beams=3
        )
        
        # Decode response
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

def evaluate_pose_description(caption):
    """Evaluates the detailed description to check if the dog is standing in side-profile."""
    caption_lower = caption.lower()
    
    # 1. Reject negative poses, angles, or close-ups
    negative_keywords = [
        "sitting", "sit ", "sits ", "lay ", "lays ", "lying", "laying", "resting", "rest ", "sleep", "curled",
        "close-up", "close up", "portrait", "headshot", "head of a", "face of a", "front view", "rear view", 
        "looking directly at the camera", "looking at the camera", "facing the camera", "facing front", "front profile"
    ]
    for neg in negative_keywords:
        if neg in caption_lower:
            return False, f"rejected due to negative keyword: '{neg}'"
            
    # 2. Enforce standing/walking pose
    positive_poses = ["standing", "stand ", "stands ", "walk", "trot", "run"]
    has_positive_pose = any(pos in caption_lower for pos in positive_poses)
    if not has_positive_pose:
        return False, "rejected due to missing standing/walking pose keywords"
        
    # 3. Enforce side-profile view (left, right, profile, side view)
    side_keywords = ["side", "profile", "left", "right", "from the side"]
    has_side = any(side in caption_lower for side in side_keywords)
    if not has_side:
        return False, "rejected due to missing side profile view keywords"
        
    return True, "passed"

def run_side_view_filtering(state, state_file):
    """Filters cropped images to keep only full-body standing side profiles using caption analysis."""
    logger.info("Starting side-view filtering stage (Florence-2)...")
    
    # Initialize the Florence filter
    vlm = FlorenceFilter()
    
    # Intermediate filtered directory
    os.makedirs(config.FILTERED_DIR, exist_ok=True)
    
    stats = {
        "total_processed": 0,
        "skipped_resume": 0,
        "passed": 0,
        "rejected": 0,
        "errors": 0
    }
    
    for breed in config.BREEDS.keys():
        cropped_breed_dir = os.path.join(config.CROPPED_DIR, breed)
        filtered_breed_dir = os.path.join(config.FILTERED_DIR, breed)
        
        if not os.path.exists(cropped_breed_dir):
            logger.warning(f"Cropped directory does not exist for {breed}. Skipping.")
            continue
            
        os.makedirs(filtered_breed_dir, exist_ok=True)
        filenames = [f for f in os.listdir(cropped_breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if not filenames:
            logger.warning(f"No cropped images found for {breed}.")
            continue
            
        logger.info(f"Filtering {len(filenames)} cropped images for {breed}...")
        
        for filename in tqdm(filenames, desc=f"Filtering {breed}", unit="img"):
            crop_path = os.path.join(cropped_breed_dir, filename)
            dest_path = os.path.join(filtered_breed_dir, filename)
            
            # Resume check
            if filename in state["filtered"]:
                saved_status = state["filtered"][filename].get("status")
                if saved_status == "passed" and os.path.exists(dest_path):
                    stats["skipped_resume"] += 1
                    stats["passed"] += 1
                    continue
                elif saved_status != "passed":
                    stats["skipped_resume"] += 1
                    stats["rejected"] += 1
                    continue
            
            stats["total_processed"] += 1
            
            try:
                # Open image
                with Image.open(crop_path) as img:
                    img = img.convert("RGB")
                    
                    # Generate detailed visual caption
                    caption = vlm.generate_caption(img)
                
                # Check description rules
                is_passed, reason = evaluate_pose_description(caption)
                
                if is_passed:
                    # Copy to filtered directory
                    with Image.open(crop_path) as img:
                        img.save(dest_path, "JPEG", quality=95)
                        
                    state["filtered"][filename] = {
                        "status": "passed",
                        "caption": caption,
                        "filtered_path": os.path.join("filtered", breed, filename)
                    }
                    stats["passed"] += 1
                    logger.info(f"Passed: {filename} -> {caption}")
                else:
                    state["filtered"][filename] = {
                        "status": "rejected",
                        "caption": caption,
                        "reason": reason
                    }
                    stats["rejected"] += 1
                    logger.debug(f"Rejected: {filename} ({reason}) -> {caption}")
                    
                save_state(state, state_file)
                
            except Exception as e:
                logger.error(f"Error filtering image {filename} with Florence-2: {e}")
                stats["errors"] += 1
                state["filtered"][filename] = {
                    "status": "error",
                    "error": str(e)
                }
                save_state(state, state_file)
                
    logger.info(
        f"Florence-2 Stage Finished. "
        f"Processed: {stats['total_processed']}, "
        f"Passed: {stats['passed']}, "
        f"Rejected: {stats['rejected']}, "
        f"Errors/Skipped: {stats['errors'] + stats['skipped_resume']}"
    )
    
    return stats
