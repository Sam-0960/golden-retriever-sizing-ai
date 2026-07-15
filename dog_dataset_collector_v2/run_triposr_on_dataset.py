#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from PIL import Image

# Directories
DATASET_V2_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DATASET_V2_DIR, "dataset")
OUTPUT_3D_DIR = os.path.join(DATASET_V2_DIR, "dataset_3d")

# Configure this to the path of your local TripoSR installation directory.
# You can also override the default by setting the TRIPOSR_DIR environment variable.
TRIPOSR_DIR = os.environ.get(
    "TRIPOSR_DIR",
    os.path.expanduser("~/TripoSR")
)
TRIPOSR_PYTHON = os.path.join(TRIPOSR_DIR, ".venv/bin/python")

def main():
    print("==================================================")
    print("      TripoSR 3D Dog Reconstruction Pipeline       ")
    print("==================================================")
    
    # Check if TripoSR folder exists
    if not os.path.exists(TRIPOSR_DIR) or not os.path.exists(TRIPOSR_PYTHON):
        print(f"Error: Could not find TripoSR installation or its virtual environment at: {TRIPOSR_DIR}")
        print("Set the TRIPOSR_DIR environment variable to your local TripoSR path, e.g.:")
        print("  export TRIPOSR_DIR=~/path/to/TripoSR")
        sys.exit(1)
        
    # Check if dataset exists and has breed folders
    breeds = ["golden", "labrador"]
    selected_images = {}
    
    for breed in breeds:
        breed_dir = os.path.join(DATASET_DIR, breed)
        if not os.path.exists(breed_dir):
            print(f"Error: Breed directory not found: {breed_dir}")
            sys.exit(1)
            
        images = sorted([f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not images:
            print(f"Warning: No images found in: {breed_dir}")
            continue
            
        # Select first 10 images for each breed
        selected_images[breed] = images[:10]
        print(f"Selected for '{breed}': {', '.join(images[:10])}")
        
    if not selected_images:
        print("Error: No images found to reconstruct.")
        sys.exit(1)
        
    os.makedirs(OUTPUT_3D_DIR, exist_ok=True)
    
    # We will invoke a subprocess using TripoSR's virtual environment python to run background removal and TripoSR.
    # This ensures that rembg and PyTorch are loaded from the environment that has them fully configured.
    for breed, files in selected_images.items():
        for filename in files:
            image_path = os.path.join(DATASET_DIR, breed, filename)
            name_slug = f"{breed}_{os.path.splitext(filename)[0]}"
            
            print(f"\nProcessing 3D reconstruction for: {name_slug}...")
            
            # Temporary no-bg image path
            nobg_path = os.path.join(OUTPUT_3D_DIR, f"{name_slug}_nobg.png")
            
            # 1. Remove Background
            print("  → Removing background using rembg...")
            # We call python with a script string to run rembg from the TripoSR environment
            rembg_script = f"""
from PIL import Image
from rembg import remove
input_img = Image.open("{image_path}").convert("RGBA")
output_img = remove(input_img)
output_img.save("{nobg_path}")
"""
            res = subprocess.run([TRIPOSR_PYTHON, "-c", rembg_script], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  ✗ Background removal failed:\n{res.stderr}")
                continue
            print(f"  ✓ Saved nobg image: {nobg_path}")
            
            # 2. Run TripoSR
            print("  → Running TripoSR reconstruction on MPS...")
            temp_output_dir = os.path.join(OUTPUT_3D_DIR, f"temp_output_{name_slug}")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            triposr_cmd = [
                TRIPOSR_PYTHON, "run.py", nobg_path,
                "--device", "mps",
                "--dtype", "float32",
                "--model-save-format", "glb",
                "--output-dir", temp_output_dir
            ]
            
            res = subprocess.run(triposr_cmd, cwd=TRIPOSR_DIR, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  ✗ TripoSR reconstruction failed:\n{res.stderr}")
                # Clean up temp folder
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                continue
                
            # Locate output glb
            glb_path = os.path.join(temp_output_dir, "0", "mesh.glb")
            if not os.path.exists(glb_path):
                print(f"  ✗ mesh.glb not found at {glb_path}")
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                continue
                
            # Save final GLB to OUTPUT_3D_DIR
            final_glb_path = os.path.join(OUTPUT_3D_DIR, f"{name_slug}_mesh.glb")
            shutil.copy2(glb_path, final_glb_path)
            print(f"  ✓ 3D Mesh successfully saved to: {final_glb_path}")
            
            # Clean up temp directories
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            if os.path.exists(nobg_path):
                os.remove(nobg_path)
                
    print("\n==================================================")
    print("           Reconstruction Complete                ")
    print(f"All 3D mesh GLB files are saved in: {OUTPUT_3D_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()
