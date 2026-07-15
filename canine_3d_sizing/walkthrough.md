# Stable Fast 3D End-to-End Pipeline Walkthrough

We have successfully created a unified, 100% offline pipeline script ([run_pipeline.py](file:///N:/PES/CCBD/Internship/Stable-fast/run_pipeline.py)) that integrates foreground segmenting, safety cropping, camera depth perception map rendering, 3D reconstruction, and calibrated physical dimension extraction.

---

## 1. Pipeline Execution Flow
When you run the pipeline:
1. **Dynamic Silhouette Cropping**: The script isolates the dog's boundary using `rembg` and crops the original image tightly, adding a safe **10% padding margin** to ensure the entire tail, ears, and paws are fully visible and preserved.
2. **3D Reconstruction**: Spawns a Stable Fast 3D run on the cropped image to generate the 3D model.
3. **Camera Depth Perception Map Rendering**: Using our custom offline Z-buffer rasterizer, the script projects the 3D mesh back onto the camera viewplane and renders a perfectly pixel-aligned greyscale depth map (`depth_perception.png`). This requires no internet connection and captures the exact depth profile of the dog relative to the camera focal point.
4. **Calibrated Sizing Extraction**: Translates and rotates the dog mesh in 3D space, fits a quadratic polynomial curve to smooth out spine wiggles, and scales all dimensions using the rotation-invariant 3D Spine Back Length (default: 55.0 cm). Girths are calculated using veterinary width-to-depth aspect ratio priors.

---

## 2. Outputs Generated
All outputs for the photo are saved under a dedicated subfolder in your workspace:
📂 **[pipeline_outputs/dog_indoor_standing/](file:///N:/PES/CCBD/Internship/Stable-fast/pipeline_outputs/dog_indoor_standing)**
- [input_cropped.jpg](file:///N:/PES/CCBD/Internship/Stable-fast/pipeline_outputs/dog_indoor_standing/input_cropped.jpg) (Cropped RGB image with 10% safety margin)
- [cropped_dog.png](file:///N:/PES/CCBD/Internship/Stable-fast/pipeline_outputs/dog_indoor_standing/cropped_dog.png) (Transparent PNG of the isolated dog)
- [depth_perception.png](file:///N:/PES/CCBD/Internship/Stable-fast/pipeline_outputs/dog_indoor_standing/depth_perception.png) (Greyscale camera depth perception map)
- [mesh.glb](file:///N:/PES/CCBD/Internship/Stable-fast/pipeline_outputs/dog_indoor_standing/mesh.glb) (Reconstructed 3D mesh model)
- [measurements.txt](file:///N:/PES/CCBD/Internship/Stable-fast/pipeline_outputs/dog_indoor_standing/measurements.txt) (Extracted sizing dimensions and recommended apparel sizes)

---

## 3. Extracted Sizing Results (`measurements.txt`)
*Calibrated to a standard **55.0 cm** Back Length (3D Spine):*

- **Anchor Back Length**: 55.0 cm
- **Estimated Height**: **51.6 cm** (Matches previous standing runs!)
- **Estimated Chest Girth**: **60.1 cm** (Matches previous standing runs!)
- **Estimated Neck Girth**: **55.7 cm** (Matches previous standing runs!)
- **Recommended Harness Size**: **Medium** (Chest: 56–69 cm range)
- **Recommended VTon Clothing Size**: **Large** (Standard Retriever fit)
