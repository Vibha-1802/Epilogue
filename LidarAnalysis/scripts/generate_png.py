import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import argparse
import sys
import os

def generate_visualization(frame_id):
    # Load the frame with proper zero-padding (e.g. 7 -> frame_00007.pcd)
    filename = f"frame_{frame_id:05d}.pcd"
    file_path = f"../data/raw/lidar_with_intensity_and_clusters/{filename}"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find file {file_path}")
        sys.exit(1)
        
    t_pcd = o3d.t.io.read_point_cloud(file_path)
    
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    class_ids = t_pcd.point.class_id.numpy()[valid_mask].flatten()
    
    # Initialize colors
    colors = np.zeros((len(valid_points), 3))
    
    # Define mapping according to user's legend request
    mask_dynamic = np.isin(class_ids, [1, 2, 3, 4])
    mask_static = np.isin(class_ids, [6])
    mask_terrain = np.isin(class_ids, [7, 8])
    mask_others = np.isin(class_ids, [0, 5])
    
    # Colors for Open3D (0.0 to 1.0)
    c_red = np.array([255, 77, 77]) / 255.0
    c_purple = np.array([153, 50, 204]) / 255.0
    c_cyan = np.array([0, 191, 255]) / 255.0
    c_green = np.array([60, 179, 113]) / 255.0
    
    colors[mask_dynamic] = c_red
    colors[mask_static] = c_purple
    colors[mask_terrain] = c_cyan
    colors[mask_others] = c_green
    
    # Setup Open3D visualization exactly like the user's screenshot
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Capture", width=1920, height=1080, visible=True)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd_vis)
    
    # EXACT camera params from notebook to get the ego vehicle perspective
    ctr = vis.get_view_control()
    ctr.set_lookat([10.0, 0.0, 0.0])
    ctr.set_front([-1.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.05)
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.0) # Let window fully initialize on mac
    
    vis.update_geometry(pcd_vis)
    ctr.set_lookat([10.0, 0.0, 0.0])
    ctr.set_front([-1.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.05)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the rendered image from Open3D
    img = vis.capture_screen_float_buffer(do_render=True)
    img_np = (np.asarray(img) * 255.0).astype(np.uint8)
    vis.destroy_window()
    
    # Now use matplotlib to overlay the legend on top of the rendered image
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    
    # Display the Open3D rendered image
    ax.imshow(img_np)
    ax.axis('off') # Hide axes
    
    # Create custom legend matching the user's provided image
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Dynamic\\n(e.g., pedestrian)', 
               markerfacecolor=c_red, markersize=14),
        Line2D([0], [0], marker='o', color='w', label='Static\\n(e.g., pole, vehicle)', 
               markerfacecolor=c_purple, markersize=14),
        Line2D([0], [0], marker='o', color='w', label='Terrain\\n(e.g., road, ground)', 
               markerfacecolor=c_cyan, markersize=14),
        Line2D([0], [0], marker='o', color='w', label='Others\\n(e.g., vegetation)', 
               markerfacecolor=c_green, markersize=14)
    ]
    
    # Add legend to the plot in the upper left corner
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, 
              facecolor='#f0f8ff', edgecolor='#99ccff', borderpad=1.5, labelspacing=1.5)
    
    plt.tight_layout(pad=0)
    plt.savefig('classified_frame_visualization.png', bbox_inches='tight', pad_inches=0.0)
    print("Successfully saved classified_frame_visualization.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LiDAR PNG visualization")
    parser.add_argument("--frame", type=int, default=10, help="Frame number to visualize (e.g., 7 for frame_00007.pcd)")
    args = parser.parse_args()
    
    generate_visualization(args.frame)
