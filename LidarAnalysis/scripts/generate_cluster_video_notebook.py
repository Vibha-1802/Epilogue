import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# LiDAR PCD with Clusters: Video Playback
This notebook reads Point Cloud Data (PCD) files that include **intensity and cluster information** (such as `actor_id` or `class_id`) and visualizes them sequentially like a video.

We will use Open3D's Tensor API to load the point clouds, as it natively handles custom attributes. We will map the cluster IDs to distinct colors."""

code1 = """import open3d as o3d
import numpy as np
import glob
import os
import time
import matplotlib.pyplot as plt

print("Open3D:", o3d.__version__)
print("Numpy:", np.__version__)"""

text2 = """## Setting up the Video Playback
We will gather all the PCD files in the `lidar_with_intensity_and_clusters` directory, sort them to maintain frame order, and prepare the Open3D visualizer.

You can adjust the `PLAYBACK_SPEED` variable below. A value of `1.0` plays at 10 FPS (the capture rate of the simulation), `0.5` plays at half speed (5 FPS), and `2.0` plays at double speed (20 FPS)."""

code2 = """# Set playback speed multiplier (1.0 = normal 10fps, 0.5 = half speed, 2.0 = double speed)
PLAYBACK_SPEED = 1.0

# Check for lidar_with_intensity_and_clusters in current directory or parent directory
data_dir = "lidar_with_intensity_and_clusters"
if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", "lidar_with_intensity_and_clusters")):
    data_dir = os.path.join("..", "lidar_with_intensity_and_clusters")
    
pcd_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.pcd")))

if not pcd_files:
    print(f"Error: No PCD files found in {data_dir}")
else:
    print(f"Found {len(pcd_files)} frames. Ready for playback.")"""

text3 = """## Frame Processing Function
We define a helper function to read a single frame, filter out invalid points (`NaN`/`Inf`), and assign colors based on the cluster `class_id`. Points with a `class_id` of 0 (background) will be colored using their intensity, while other classes will get distinct colors."""

code3 = """def process_cluster_frame(file_path):
    t_pcd = o3d.t.io.read_point_cloud(file_path)
    
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    
    valid_points = points[valid_mask]
    
    # Check for intensity and class_id
    if 'intensity' in t_pcd.point:
        intensities = t_pcd.point.intensity.numpy()[valid_mask].flatten()
    else:
        intensities = np.zeros(len(valid_points))
        
    if 'class_id' in t_pcd.point:
        class_ids = t_pcd.point.class_id.numpy()[valid_mask].flatten()
    else:
        class_ids = np.zeros(len(valid_points))
        
    # Default color: grayscale based on intensity
    intensity_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    cmap = plt.get_cmap('gray')
    colors = cmap(intensity_norm)[:, :3]
    
    # Overlay colors for distinct classes (e.g. 1 = car, 6 = guardrail, etc.)
    # We use a categorical colormap like 'tab20'
    cluster_cmap = plt.get_cmap('tab20')
    
    # Assuming class_ids > 0 are objects of interest
    object_mask = class_ids > 0
    if np.any(object_mask):
        # Map class_id to distinct colors, modulo the colormap size
        unique_classes = np.unique(class_ids[object_mask])
        for cid in unique_classes:
            cid_int = int(cid) % 20
            # Get RGB from the categorical colormap
            obj_color = cluster_cmap(cid_int)[:3]
            colors[class_ids == cid] = obj_color
            
    return valid_points, colors
"""

text4 = """## Run the Video Visualization
This will open an external Open3D window. We will lock the camera perspective to the Ego-Vehicle (the sensor origin) and iterate through the frames.

**Note:** The visualization window will pop up externally. If it seems to freeze or jitter, make sure the cell finishes executing before trying to interact heavily."""

code4 = """if pcd_files:
    print("Initializing first frame...")
    valid_points, colors = process_cluster_frame(pcd_files[0])
    
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Cluster Playback", width=1280, height=720)
    vis.add_geometry(pcd_vis)
    
    # Configure Ego-Perspective Camera
    ctr = vis.get_view_control()
    ctr.set_lookat([10.0, 0.0, 0.0])
    ctr.set_front([-1.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.05) 
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.0)
    
    print("Starting playback...")
    for i in range(1, len(pcd_files)):
        start_time = time.time()
        
        valid_points, colors = process_cluster_frame(pcd_files[i])
        
        pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        
        vis.update_geometry(pcd_vis)
        
        # Enforce camera view
        ctr.set_lookat([10.0, 0.0, 0.0])
        ctr.set_front([-1.0, 0.0, 0.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.05)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Target duration: 10 FPS base speed * playback speed modifier
        target_frame_duration = 0.1 / PLAYBACK_SPEED
        
        elapsed = time.time() - start_time
        sleep_time = max(target_frame_duration - elapsed, 0)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    vis.destroy_window()
    print("Playback finished.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4)
]

with open('04_Cluster_Video_Visualization.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created 04_Cluster_Video_Visualization.ipynb")
