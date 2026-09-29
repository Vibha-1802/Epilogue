import open3d as o3d
import numpy as np
import glob
import os

data_dir = "../data/raw/lidar_with_intensity_and_clusters"
pcd_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.pcd")))

best_frame = None
min_combined_dist = float('inf')

for fp in pcd_files:
    t_pcd = o3d.t.io.read_point_cloud(fp)
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    
    if 'class_id' in t_pcd.point:
        class_ids = t_pcd.point.class_id.numpy()[valid_mask].flatten()
        valid_points = points[valid_mask]
        
        # dynamic: 1,2,3,4
        mask_dynamic = np.isin(class_ids, [1, 2, 3, 4])
        # guardrail: 6
        mask_guardrail = (class_ids == 6)
        
        if np.any(mask_dynamic) and np.any(mask_guardrail):
            # calculate distance to origin
            dist_dynamic = np.linalg.norm(valid_points[mask_dynamic][:, :2], axis=1)
            dist_guardrail = np.linalg.norm(valid_points[mask_guardrail][:, :2], axis=1)
            
            min_d_dyn = np.min(dist_dynamic)
            min_d_guard = np.min(dist_guardrail)
            
            score = min_d_dyn + min_d_guard
            
            if score < min_combined_dist:
                min_combined_dist = score
                best_frame = fp
                print(f"New best: {os.path.basename(fp)} - Dynamic Dist: {min_d_dyn:.2f}, Guardrail Dist: {min_d_guard:.2f}")
                
print(f"\\nFinal Best Frame: {best_frame}")
