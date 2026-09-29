import open3d as o3d
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import time

def process_cluster_frame(file_path):
    t_pcd = o3d.t.io.read_point_cloud(file_path)
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    
    if 'intensity' in t_pcd.point:
        intensities = t_pcd.point.intensity.numpy()[valid_mask].flatten()
    else:
        intensities = np.zeros(len(valid_points))
        
    if 'class_id' in t_pcd.point:
        class_ids = t_pcd.point.class_id.numpy()[valid_mask].flatten()
    else:
        class_ids = np.zeros(len(valid_points))
        
    intensity_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    cmap = plt.get_cmap('gray')
    colors = cmap(intensity_norm)[:, :3]
    
    cluster_cmap = plt.get_cmap('tab20')
    object_mask = class_ids > 0
    if np.any(object_mask):
        unique_classes = np.unique(class_ids[object_mask])
        for cid in unique_classes:
            cid_int = int(cid) % 20
            obj_color = cluster_cmap(cid_int)[:3]
            colors[class_ids == cid] = obj_color
            
    return valid_points, colors

def main():
    data_dir = "../data/raw/lidar_with_intensity_and_clusters"
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.pcd")))
    
    if not pcd_files:
        print(f"Error: No PCD files found in {data_dir}")
        return

    print(f"Found {len(pcd_files)} frames. Generating Open3D MP4...")

    WIDTH = 1280
    HEIGHT = 720
    PLAYBACK_SPEED = 0.2
    BASE_FPS = 10
    OUTPUT_FPS = BASE_FPS * PLAYBACK_SPEED

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('lidar_playback.mp4', fourcc, OUTPUT_FPS, (WIDTH, HEIGHT))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR MP4 Export", width=WIDTH, height=HEIGHT, visible=True)
    
    # White background like the notebook
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    
    pcd_vis = o3d.geometry.PointCloud()
    
    # Process first frame to initialize geometry
    valid_points, colors = process_cluster_frame(pcd_files[0])
    pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd_vis)

    # EXACT camera params from notebook
    ctr = vis.get_view_control()
    ctr.set_lookat([10.0, 0.0, 0.0])
    ctr.set_front([-1.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.05) 
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.0) # Let window fully initialize on mac

    # Capture frames
    for i, file_path in enumerate(pcd_files):
        print(f"Rendering frame {i+1}/{len(pcd_files)}...", end="\\r")
        valid_points, colors = process_cluster_frame(file_path)
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
        
        # Capture screen image
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        img = (img * 255.0).astype(np.uint8)
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if img_bgr.shape[1] != WIDTH or img_bgr.shape[0] != HEIGHT:
            img_bgr = cv2.resize(img_bgr, (WIDTH, HEIGHT))
            
        out_video.write(img_bgr)

    out_video.release()
    vis.destroy_window()
    print("\\nMP4 successfully created: lidar_playback.mp4")

if __name__ == "__main__":
    main()
