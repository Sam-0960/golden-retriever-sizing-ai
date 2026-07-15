import os
import sys
import time
import argparse
import subprocess
import numpy as np
import trimesh
from PIL import Image

# Setup standard path configurations
PYTHON_PATH = r"C:\Users\Kruthik\miniconda3\envs\sf3d\python.exe"
WORKSPACE_DIR = r"N:\PES\CCBD Internship\Stable-fast"
OUTPUT_BASE_DIR = os.path.join(WORKSPACE_DIR, "pipeline_outputs")

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def crop_dog_image(input_path, output_dir):
    """
    Isolate the dog using rembg, crop tightly with a safe 10% margin,
    and save both the cropped RGB image and transparent RGBA image.
    """
    import rembg
    
    print("Step 1: Segmenting and cropping foreground...")
    img = Image.open(input_path).convert("RGBA")
    
    # Extract alpha mask
    rembg_session = rembg.new_session()
    out_rgba = rembg.remove(img, session=rembg_session)
    alpha = np.array(out_rgba)[:, :, 3]
    
    # Bounding box coordinates of foreground
    non_zero = np.argwhere(alpha > 0)
    if len(non_zero) > 0:
        y_min, x_min = non_zero.min(axis=0)
        y_max, x_max = non_zero.max(axis=0)
        
        # Add a safe 10% padding margin to ensure full dog is visible
        h, w = img.height, img.width
        pad_x = int((x_max - x_min) * 0.10)
        pad_y = int((y_max - y_min) * 0.10)
        
        x_min_pad = max(0, x_min - pad_x)
        y_min_pad = max(0, y_min - pad_y)
        x_max_pad = min(w, x_max + pad_x)
        y_max_pad = min(h, y_max + pad_y)
        
        # Crop RGBA (transparent) and RGB (flat background)
        cropped_rgba = out_rgba.crop((x_min_pad, y_min_pad, x_max_pad, y_max_pad))
        cropped_rgb = img.convert("RGB").crop((x_min_pad, y_min_pad, x_max_pad, y_max_pad))
        
        cropped_rgba_path = os.path.join(output_dir, "cropped_dog.png")
        cropped_rgb_path = os.path.join(output_dir, "input_cropped.jpg")
        
        cropped_rgba.save(cropped_rgba_path)
        cropped_rgb.save(cropped_rgb_path)
        print(f"  Cropped dog saved to {cropped_rgb_path}")
        return cropped_rgb_path, cropped_rgba_path
    else:
        # Fallback to saving original if no foreground detected
        print("  Warning: No dog detected by rembg, using original image.")
        cropped_rgb_path = os.path.join(output_dir, "input_cropped.jpg")
        img.convert("RGB").save(cropped_rgb_path)
        return cropped_rgb_path, None

def generate_offline_depth_map(mesh_path, output_dir, out_resolution=512):
    """
    Perform camera depth perception by rasterizing the reconstructed 3D mesh GLB
    projected back onto the camera viewplane using a custom Z-buffer renderer.
    """
    print("Step 2: Rendering camera depth perception map...")
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in mesh.geometry.values()
        ])
        
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y
    
    u = ((x - x_min) / (x_max - x_min) * (out_resolution - 1)).astype(np.float32)
    v = ((1.0 - (y - y_min) / (y_max - y_min)) * (out_resolution - 1)).astype(np.float32)
    
    # Initialize Z-buffer with positive infinity
    z_buffer = np.full((out_resolution, out_resolution), np.inf, dtype=np.float32)
    
    # Software Z-buffer rasterizer loop
    for face in faces:
        u0, u1, u2 = u[face[0]], u[face[1]], u[face[2]]
        v0, v1, v2 = v[face[0]], v[face[1]], v[face[2]]
        z0, z1, z2 = z[face[0]], z[face[1]], z[face[2]]
        
        min_u = int(max(0, min(u0, u1, u2)))
        max_u = int(min(out_resolution - 1, max(u0, u1, u2)))
        min_v = int(max(0, min(v0, v1, v2)))
        max_v = int(min(out_resolution - 1, max(v0, v1, v2)))
        
        if min_u > max_u or min_v > max_v:
            continue
            
        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denom) < 1e-6:
            continue
            
        for py in range(min_v, max_v + 1):
            for px in range(min_u, max_u + 1):
                w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) / denom
                w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) / denom
                w2 = 1.0 - w0 - w1
                
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    pz = w0 * z0 + w1 * z1 + w2 * z2
                    if pz < z_buffer[py, px]:
                        z_buffer[py, px] = pz
                        
    # Normalize depth map
    mask = np.isfinite(z_buffer)
    depth_map = np.zeros((out_resolution, out_resolution), dtype=np.uint8)
    if mask.sum() > 0:
        z_min_val = z_buffer[mask].min()
        z_max_val = z_buffer[mask].max()
        if z_max_val > z_min_val:
            # Map closer points (smaller Z) to brighter white (255), farther points to darker grey (25)
            depth_map[mask] = (255 - (z_buffer[mask] - z_min_val) / (z_max_val - z_min_val) * 230).astype(np.uint8)
        else:
            depth_map[mask] = 255
            
    depth_map_path = os.path.join(output_dir, "depth_perception.png")
    Image.fromarray(depth_map, mode="L").save(depth_map_path)
    print(f"  Depth perception map saved to {depth_map_path}")
    return depth_map_path

def run_3d_model(cropped_rgb_path, output_dir):
    """
    Run Stable Fast 3D on the cropped photo.
    """
    print("Step 3: Generating 3D mesh model...")
    mesh_output_dir = os.path.join(output_dir, "mesh_run")
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    cmd = [
        PYTHON_PATH, "run.py", cropped_rgb_path, "--output-dir", mesh_output_dir
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    src_mesh = os.path.join(mesh_output_dir, "0", "mesh.glb")
    dest_mesh = os.path.join(output_dir, "mesh.glb")
    
    if os.path.exists(src_mesh):
        import shutil
        shutil.move(src_mesh, dest_mesh)
        shutil.rmtree(mesh_output_dir)
        print(f"  3D model saved to {dest_mesh}")
        return dest_mesh
    else:
        raise FileNotFoundError("Stable Fast 3D failed to generate mesh.glb")

def rotate_y(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return pts @ R.T

def rotate_z(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    return pts @ R.T

def extract_aligned_measurements(mesh_path, output_dir, KNOWN_BACK_LENGTH_CM=55.0):
    """
    Translate, rotate (yaw/pitch), smooth, and calculate size metrics
    from the 3D mesh using anatomical aspect ratio priors.
    """
    print("Step 4: Extracting calibrated physical measurements...")
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in mesh.geometry.values()
        ])
        
    vertices = mesh.vertices.copy()
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    total_length_raw = x_max - x_min
    
    num_slices = 100
    x_step = total_length_raw / num_slices
    slice_info = []
    
    for i in range(num_slices):
        x_low = x_min + i * x_step
        x_high = x_low + x_step
        mask = (vertices[:, 0] >= x_low) & (vertices[:, 0] < x_high)
        slice_pts = vertices[mask]
        
        if len(slice_pts) > 10:
            slice_info.append({
                "x": (x_low + x_high) / 2,
                "y_max": slice_pts[:, 1].max(),
                "y_min": slice_pts[:, 1].min(),
                "z_mean": slice_pts[:, 2].mean()
            })
            
    # Find head peak
    front_half_slices = [s for s in slice_info if s["x"] < (x_min + x_max)/2]
    head_slice = max(front_half_slices, key=lambda s: s["y_max"])
    head_x = head_slice["x"]
    
    # Neck and tail base
    neck_slice = [s for s in slice_info if s["x"] > head_x][5]
    neck_raw = np.array([neck_slice["x"], neck_slice["y_max"], neck_slice["z_mean"]])
    
    raw_height = y_max - y_min
    tail_base_slice = None
    for s in reversed(slice_info):
        slice_h = s["y_max"] - s["y_min"]
        if s["x"] > head_x and slice_h > 0.25 * raw_height:
            tail_base_slice = s
            break
    if tail_base_slice is None:
        tail_base_slice = slice_info[-10]
    tail_raw = np.array([tail_base_slice["x"], tail_base_slice["y_max"], tail_base_slice["z_mean"]])
    
    # 3D Alignment Rotation
    vertices = vertices - neck_raw
    tail_centered = tail_raw - neck_raw
    
    yaw = np.arctan2(tail_centered[2], tail_centered[0])
    vertices = rotate_y(vertices, -yaw)
    tail_yaw = rotate_y(tail_centered.reshape(1, 3), -yaw)[0]
    
    pitch = np.arctan2(tail_yaw[1], tail_yaw[0])
    vertices = rotate_z(vertices, -pitch)
    
    # Recalculate aligned bounds
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    total_length_aligned = x_max - x_min
    
    # Refresh aligned slices
    x_step = total_length_aligned / num_slices
    aligned_slices = []
    for i in range(num_slices):
        x_low = x_min + i * x_step
        x_high = x_low + x_step
        mask = (vertices[:, 0] >= x_low) & (vertices[:, 0] < x_high)
        slice_pts = vertices[mask]
        if len(slice_pts) > 10:
            aligned_slices.append({
                "x": (x_low + x_high) / 2,
                "y_max": slice_pts[:, 1].max(),
                "y_min": slice_pts[:, 1].min()
            })
            
    aligned_neck_x = 0.0
    aligned_chest_x = aligned_neck_x + 0.12 * total_length_aligned
    
    aligned_tail_base_slice = None
    for s in reversed(aligned_slices):
        slice_h = s["y_max"] - s["y_min"]
        if s["x"] > aligned_neck_x and slice_h > 0.25 * (y_max - y_min):
            aligned_tail_base_slice = s
            break
    if aligned_tail_base_slice is None:
        aligned_tail_base_slice = aligned_slices[-10]
    aligned_tail_base_x = aligned_tail_base_slice["x"]
    
    # Trace spine and fit polynomial
    spine_x_vals = []
    spine_y_vals = []
    for s in aligned_slices:
        if aligned_neck_x <= s["x"] <= aligned_tail_base_x:
            mask = (vertices[:, 0] >= s["x"] - x_step/2) & (vertices[:, 0] < s["x"] + x_step/2)
            pts_in_slice = vertices[mask]
            if len(pts_in_slice) > 0:
                top_y = pts_in_slice[:, 1].max()
                spine_x_vals.append(s["x"])
                spine_y_vals.append(top_y)
                
    poly_coeffs = np.polyfit(spine_x_vals, spine_y_vals, 2)
    poly_fit = np.poly1d(poly_coeffs)
    
    smooth_x = np.linspace(aligned_neck_x, aligned_tail_base_x, 100)
    smooth_y = poly_fit(smooth_x)
    
    back_length_raw = 0.0
    for k in range(len(smooth_x) - 1):
        dx = smooth_x[k+1] - smooth_x[k]
        dy = smooth_y[k+1] - smooth_y[k]
        back_length_raw += np.sqrt(dx**2 + dy**2)
        
    scale_factor = KNOWN_BACK_LENGTH_CM / back_length_raw
    height = (y_max - y_min) * scale_factor
    back_length = KNOWN_BACK_LENGTH_CM
    
    def compute_anatomical_girth(target_x, is_neck=False, tolerance_pct=0.03):
        tol = total_length_aligned * tolerance_pct
        pts = vertices[np.abs(vertices[:, 0] - target_x) < tol]
        if len(pts) == 0:
            return 0.0
        y_slice_min, y_slice_max = pts[:, 1].min(), pts[:, 1].max()
        
        if is_neck:
            depth_raw = y_slice_max - y_slice_min
        else:
            cutoff = y_slice_min + 0.45 * (y_slice_max - y_slice_min)
            body_pts = pts[pts[:, 1] > cutoff]
            if len(body_pts) == 0:
                body_pts = pts
            depth_raw = body_pts[:, 1].max() - body_pts[:, 1].min()
            
        depth = depth_raw * scale_factor
        width = 0.80 * depth if is_neck else 0.72 * depth
        
        a, b = width / 2, depth / 2
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        return perimeter

    chest_girth = compute_anatomical_girth(aligned_chest_x, is_neck=False)
    neck_girth = compute_anatomical_girth(aligned_neck_x, is_neck=True)
    
    # Save output text file
    txt_path = os.path.join(output_dir, "measurements.txt")
    with open(txt_path, "w") as f:
        f.write("=== Dog Physical Size Measurements ===\n")
        f.write(f"Anchor Back Length (3D Spine): {back_length:.1f} cm\n")
        f.write(f"Estimated Height: {height:.1f} cm\n")
        f.write(f"Estimated Chest Girth: {chest_girth:.1f} cm\n")
        f.write(f"Estimated Neck Girth: {neck_girth:.1f} cm\n")
        f.write("\nRecommended Apparel Sizes:\n")
        
        # Sizing recommendations based on chest girth
        if chest_girth < 43:
            h_size, c_size = "XS", "Small"
        elif chest_girth < 56:
            h_size, c_size = "S", "Medium"
        elif chest_girth < 69:
            h_size, c_size = "M", "Large"
        elif chest_girth < 81:
            h_size, c_size = "L", "Large/XL"
        else:
            h_size, c_size = "XL", "XL/XXL"
            
        f.write(f"  Recommended Harness Size: {h_size}\n")
        f.write(f"  Recommended VTon Clothing Size: {c_size}\n")
        
    print(f"  Measurements saved to {txt_path}")
    print(f"    Height: {height:.1f} cm | Chest: {chest_girth:.1f} cm | Neck: {neck_girth:.1f} cm | Back: {back_length:.1f} cm")
    return txt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Dog 3D Reconstruction, Depth Perception, and Measurement Pipeline")
    parser.add_argument("image_path", type=str, help="Path to the input dog photo.")
    parser.add_argument("--back-length", type=float, default=55.0, help="Custom dog back length in cm for scaling (Default: 55.0 cm).")
    args = parser.parse_args()
    
    input_img = args.image_path
    if not os.path.exists(input_img):
        print(f"Error: Input image '{input_img}' does not exist.")
        sys.exit(1)
        
    filename = os.path.basename(input_img)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Create specific output folder for this photo
    output_dir = os.path.join(OUTPUT_BASE_DIR, name_without_ext)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"STARTING PIPELINE FOR: {filename}")
    print("="*80 + "\n")
    
    # Run Steps
    cropped_rgb, cropped_rgba = crop_dog_image(input_img, output_dir)
    mesh_path = run_3d_model(cropped_rgb, output_dir)
    depth_map = generate_offline_depth_map(mesh_path, output_dir)
    txt_path = extract_aligned_measurements(mesh_path, output_dir, KNOWN_BACK_LENGTH_CM=args.back_length)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Outputs saved in: {output_dir}")
    print("  - input_cropped.jpg    (Tightly cropped RGB)")
    print("  - cropped_dog.png      (Segmented transparent dog)")
    print("  - depth_perception.png (Greyscale depth perception map)")
    print("  - mesh.glb             (Reconstructed 3D mesh)")
    print("  - measurements.txt     (Physical dimensions & sizing info)")
    print("="*80 + "\n")
