import open3d as o3d
import numpy as np
import glob
import os

data_dir = "../lidar_dataset/Exported_PCD_Files_ryan_Final"
pcd_files = sorted(glob.glob(os.path.join(data_dir, "*.pcd")))

if not pcd_files:
    print(f"No PCD files found in {data_dir}")
    exit(1)

best_frame = None
max_score = 0

print(f"Scanning {len(pcd_files)} PCD files for an intersection...")

for fp in pcd_files:
    t_pcd = o3d.t.io.read_point_cloud(fp)
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        continue
        
    x = valid_points[:, 0]
    y = valid_points[:, 1]
    
    # An intersection has points radiating in all directions.
    # High variance in both X and Y means an open area rather than a narrow corridor.
    var_x = np.var(x)
    var_y = np.var(y)
    
    score = len(valid_points) * (var_x * var_y)
    
    if score > max_score:
        max_score = score
        best_frame = fp

print(f"Best Frame found: {best_frame}")
