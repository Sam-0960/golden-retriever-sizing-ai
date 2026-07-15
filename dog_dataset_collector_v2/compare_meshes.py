import os
import trimesh
import numpy as np

# Paths
TRIPOSR_3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_3d")
SF3D_3D_DIR = os.path.expanduser("~/Desktop/stable-fast-kruthik")

def analyze_mesh(filepath):
    try:
        # Load mesh
        mesh = trimesh.load(filepath)
        
        # If it's a Scene, concatenate geometries into a single mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in mesh.geometry.values()
            ])
            
        # Metrics
        num_verts = len(mesh.vertices)
        num_faces = len(mesh.faces)
        
        # Bounding box extents
        extents = mesh.extents
        # Volume of bounding box
        bbox_volume = np.prod(extents)
        
        # Watertight check
        is_watertight = mesh.is_watertight
        
        # File size in KB
        file_size_kb = os.path.getsize(filepath) / 1024.0
        
        return {
            "num_verts": num_verts,
            "num_faces": num_faces,
            "extents": extents.tolist(),
            "bbox_volume": bbox_volume,
            "is_watertight": is_watertight,
            "file_size_kb": file_size_kb
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    print("==================================================")
    print("          Dog 3D Mesh Comparison Utility           ")
    print("==================================================")
    
    # 1. Gather TripoSR meshes (our golden retriever meshes)
    triposr_results = {}
    print("\nAnalyzing TripoSR Golden Retriever meshes...")
    for i in range(1, 11):
        filename = f"golden_{i:05d}_mesh.glb"
        filepath = os.path.join(TRIPOSR_3D_DIR, filename)
        if os.path.exists(filepath):
            metrics = analyze_mesh(filepath)
            if metrics:
                triposr_results[filename] = metrics
                print(f"  {filename}: Verts={metrics['num_verts']}, Faces={metrics['num_faces']}, Size={metrics['file_size_kb']:.1f} KB")
        else:
            print(f"  Warning: File not found {filepath}")
            
    # 2. Gather SF3D meshes (from stable-fast-kruthik)
    sf3d_results = {}
    print("\nAnalyzing Stable Fast 3D (kruthik) meshes...")
    if os.path.exists(SF3D_3D_DIR):
        for folder in os.listdir(SF3D_3D_DIR):
            folder_path = os.path.join(SF3D_3D_DIR, folder)
            if os.path.isdir(folder_path):
                # Search for mesh.glb inside subfolders (e.g. folder/0/mesh.glb)
                glb_path = os.path.join(folder_path, "0", "mesh.glb")
                if os.path.exists(glb_path):
                    metrics = analyze_mesh(glb_path)
                    if metrics:
                        sf3d_results[folder] = metrics
                        print(f"  {folder}: Verts={metrics['num_verts']}, Faces={metrics['num_faces']}, Size={metrics['file_size_kb']:.1f} KB")
    else:
        print(f"Error: SF3D directory not found: {SF3D_3D_DIR}")
        
    # Write a summary text file
    output_summary_path = os.path.join(TRIPOSR_3D_DIR, "mesh_comparison_report.txt")
    with open(output_summary_path, "w") as f:
        f.write("==================================================\n")
        f.write("          DOG 3D MESH COMPARISON REPORT\n")
        f.write("==================================================\n\n")
        
        f.write("--- TRIPOSR GOLDEN RETRIEVER MESHES (10) ---\n")
        avg_verts_t = []
        avg_faces_t = []
        avg_size_t = []
        for name, m in triposr_results.items():
            f.write(f"{name}:\n")
            f.write(f"  Vertices:   {m['num_verts']}\n")
            f.write(f"  Faces:      {m['num_faces']}\n")
            f.write(f"  File Size:  {m['file_size_kb']:.1f} KB\n")
            f.write(f"  Watertight: {m['is_watertight']}\n")
            f.write(f"  Extents:    L={m['extents'][0]:.2f}, W={m['extents'][1]:.2f}, H={m['extents'][2]:.2f}\n\n")
            avg_verts_t.append(m['num_verts'])
            avg_faces_t.append(m['num_faces'])
            avg_size_t.append(m['file_size_kb'])
            
        f.write("--- STABLE FAST 3D (KRUTHIK) MESHES (8) ---\n")
        avg_verts_s = []
        avg_faces_s = []
        avg_size_s = []
        for name, m in sf3d_results.items():
            f.write(f"{name}:\n")
            f.write(f"  Vertices:   {m['num_verts']}\n")
            f.write(f"  Faces:      {m['num_faces']}\n")
            f.write(f"  File Size:  {m['file_size_kb']:.1f} KB\n")
            f.write(f"  Watertight: {m['is_watertight']}\n")
            f.write(f"  Extents:    L={m['extents'][0]:.2f}, W={m['extents'][1]:.2f}, H={m['extents'][2]:.2f}\n\n")
            avg_verts_s.append(m['num_verts'])
            avg_faces_s.append(m['num_faces'])
            avg_size_s.append(m['file_size_kb'])
            
        f.write("==================================================\n")
        f.write("             COMPARATIVE STATISTICS\n")
        f.write("==================================================\n")
        f.write(f"TripoSR Avg Vertices: {np.mean(avg_verts_t):.1f}\n")
        f.write(f"TripoSR Avg Faces:    {np.mean(avg_faces_t):.1f}\n")
        f.write(f"TripoSR Avg Size:     {np.mean(avg_size_t):.1f} KB\n\n")
        
        f.write(f"Stable Fast 3D (Kruthik) Avg Vertices: {np.mean(avg_verts_s):.1f}\n")
        f.write(f"Stable Fast 3D (Kruthik) Avg Faces:    {np.mean(avg_faces_s):.1f}\n")
        f.write(f"Stable Fast 3D (Kruthik) Avg Size:     {np.mean(avg_size_s):.1f} KB\n")
        
    print(f"\nWritten comparison report to: {output_summary_path}")

if __name__ == "__main__":
    main()
