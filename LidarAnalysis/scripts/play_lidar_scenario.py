import os
import glob
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def process_frame(pcd_file, cmap):
    # Load using Tensor API to handle float64 and custom attributes safely
    t_pcd = o3d.t.io.read_point_cloud(pcd_file)
    points = t_pcd.point.positions.numpy()
    
    # Check for NaN and filter invalid points
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    
    # Process intensity
    if 'intensity' in t_pcd.point:
        intensities = t_pcd.point.intensity.numpy()[valid_mask].flatten()
    else:
        intensities = np.zeros(len(valid_points))
        
    # Normalize intensity for colormap mapping
    intensity_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    colors = cmap(intensity_norm)[:, :3]
    
    return valid_points, colors

def main():
    # Set playback speed multiplier (1.0 = normal 10fps, 0.5 = half speed, 2.0 = double speed)
    PLAYBACK_SPEED = 0.2
    
    # Check for lidar_with_intensity in current directory or parent directory
    data_dir = "lidar_with_intensity"
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", "lidar_with_intensity")):
        data_dir = os.path.join("..", "lidar_with_intensity")
        
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.pcd")))
    
    if not pcd_files:
        print(f"Error: No PCD files found in {data_dir}")
        return
        
    print(f"Found {len(pcd_files)} frames. Starting playback...")
    
    # Use turbo colormap for intensity
    cmap = plt.get_cmap('turbo')
    
    # Initialize first frame
    valid_points, colors = process_frame(pcd_files[0], cmap)
    
    # Create Legacy PointCloud for Visualization
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    
    # Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Ego-Perspective Playback", width=1280, height=720)
    
    # Add the initial geometry
    vis.add_geometry(pcd_vis)
    
    # Configure Ego-Perspective Camera
    # Assume MATLAB LiDAR: +X is forward, +Y is left, +Z is up
    # Ego Sensor is essentially at the origin (0,0,0)
    ctr = vis.get_view_control()
    
    # Camera position setup:
    # lookat: the point the camera focuses on. We look exactly forward along +X.
    # front: the vector pointing FROM the lookat TO the camera. 
    #        To put the camera at origin (-X relative to lookat), the vector is [-1, 0, 0]
    # up: the up direction (+Z)
    ctr.set_lookat([10.0, 0.0, 0.0])
    ctr.set_front([-1.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    
    # Zoom out slightly for a wider Field of View
    ctr.set_zoom(0.05) 
    
    # Update renderer once to apply camera settings
    vis.poll_events()
    vis.update_renderer()
    
    # Wait a second before starting playback to let window stabilize
    time.sleep(1.0)
    
    # Playback loop
    for i in range(1, len(pcd_files)):
        start_time = time.time()
        
        valid_points, colors = process_frame(pcd_files[i], cmap)
        
        # Update point cloud data arrays
        pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        
        # Notify visualizer that geometry changed
        vis.update_geometry(pcd_vis)
        
        # Re-enforce the camera view so it stays locked to the ego-perspective
        ctr.set_lookat([10.0, 0.0, 0.0])
        ctr.set_front([-1.0, 0.0, 0.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.05)
        
        # Render the updated frame
        vis.poll_events()
        vis.update_renderer()
        
        print(f"Played frame {i+1}/{len(pcd_files)}: {os.path.basename(pcd_files[i])}")
        
        # Calculate target duration based on 10 FPS base speed * playback speed modifier
        target_frame_duration = 0.1 / PLAYBACK_SPEED
        
        # Sleep to maintain the target frame rate
        elapsed = time.time() - start_time
        sleep_time = max(target_frame_duration - elapsed, 0)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
    print("Playback finished! The window will stay open. Close the window to exit the script.")
    
    # Keep window open until user manually closes it
    vis.run() 
    vis.destroy_window()

if __name__ == "__main__":
    main()
