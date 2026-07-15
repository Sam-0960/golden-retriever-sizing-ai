import trimesh
import numpy as np
import os
import glob

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

def calculate_measurements(mesh_path, KNOWN_BACK_LENGTH_CM=55.0):
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in mesh.geometry.values()
        ])
        
    vertices = mesh.vertices.copy()
    
    # --- STEP 1: INITIAL LANDMARK IDENTIFICATION ---
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
    
    # --- STEP 2: 3D ALIGNMENT ROTATION ---
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
    
    # Refresh slices in aligned coordinate system
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
    
    # --- STEP 3: SPINE PATH CURVATURE & SCALE CALIBRATION ---
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
    
    body_length = total_length_aligned * scale_factor
    height = (y_max - y_min) * scale_factor
    back_length = KNOWN_BACK_LENGTH_CM
    
    # --- STEP 4: ANATOMICAL PRIOR GIRTH CALCULATIONS ---
    def compute_anatomical_girth(target_x, is_neck=False, tolerance_pct=0.03):
        tol = total_length_aligned * tolerance_pct
        pts = vertices[np.abs(vertices[:, 0] - target_x) < tol]
        if len(pts) == 0:
            return 0.0
            
        y_slice_min, y_slice_max = pts[:, 1].min(), pts[:, 1].max()
        
        # Legs are naturally rotated out of the neck slice, so we don't apply cutoff there
        if is_neck:
            depth_raw = y_slice_max - y_slice_min
        else:
            # Chest slice intersects front legs, apply 0.45 cutoff to isolate chest depth
            cutoff = y_slice_min + 0.45 * (y_slice_max - y_slice_min)
            body_pts = pts[pts[:, 1] > cutoff]
            if len(body_pts) == 0:
                body_pts = pts
            depth_raw = body_pts[:, 1].max() - body_pts[:, 1].min()
            
        depth = depth_raw * scale_factor
        
        # Apply priors
        if is_neck:
            width = 0.80 * depth
        else:
            width = 0.72 * depth
            
        a = width / 2
        b = depth / 2
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        return perimeter

    chest_girth = compute_anatomical_girth(aligned_chest_x, is_neck=False)
    neck_girth = compute_anatomical_girth(aligned_neck_x, is_neck=True)
    
    return {
        "length": body_length,
        "height": height,
        "chest_girth": chest_girth,
        "neck_girth": neck_girth,
        "back_length": back_length
    }

if __name__ == "__main__":
    mesh_path = os.path.join("output", "0", "mesh.glb")
    if not os.path.exists(mesh_path):
        batch_meshes = glob.glob(os.path.join("output_new2_cropped", "*", "0", "mesh.glb"))
        if len(batch_meshes) > 0:
            mesh_path = batch_meshes[0]
            
    m = calculate_measurements(mesh_path)
    print(f"File tested: {mesh_path}")
    print(f"Height: {m['height']:.1f} cm")
    print(f"Chest girth: {m['chest_girth']:.1f} cm")
    print(f"Neck girth: {m['neck_girth']:.1f} cm")
    print(f"Back length: {m['back_length']:.1f} cm")
