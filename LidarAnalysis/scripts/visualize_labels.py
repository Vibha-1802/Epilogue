import numpy as np
import open3d as o3d
import json
import argparse
import os

def visualize_pcd(pcd_path, class_map_path):
    # Load class map
    if os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            class_map_data = json.load(f)
        classes = class_map_data.get('classes', [])
        class_names = {c['class_id']: c['name'] for c in classes}
    else:
        print(f"Warning: class map not found at {class_map_path}")
        class_names = {}

    # Define a distinct colormap for each class ID (RGB values from 0 to 1)
    colormap = {
        0: [0.0, 0.0, 0.0],     # Unlabelled: Black
        1: [0.2, 0.6, 1.0],     # Car: Light Blue
        2: [0.0, 0.0, 1.0],     # Truck: Dark Blue
        3: [1.0, 0.0, 1.0],     # Bicycle: Magenta
        4: [1.0, 0.0, 0.0],     # Pedestrian: Red
        6: [1.0, 0.8, 0.0],     # Guardrail: Orange/Yellow
        7: [0.7, 0.7, 0.7],     # Road: Light Grey
        8: [0.1, 0.8, 0.1],     # Terrain: Green
    }
    default_color = [1.0, 1.0, 1.0] # White for unknown

    print(f"Loading {pcd_path}...")
    
    # Read the custom ASCII PCD file manually
    # Skipping the 11 header lines to get straight to the data
    try:
        data = np.loadtxt(pcd_path, skiprows=11)
    except Exception as e:
        print(f"Error reading {pcd_path}: {e}")
        return

    xyz = data[:, 0:3]
    class_id = data[:, 5].astype(int)

    # Filter out NaNs (Open3D fails to render them properly)
    valid_mask = ~np.isnan(xyz).any(axis=1)
    xyz = xyz[valid_mask]
    class_id = class_id[valid_mask]

    # Initialize Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Apply colors based on class_id
    colors = np.zeros((len(class_id), 3))
    
    print("\n--- Legend ---")
    for cid in np.unique(class_id):
        name = class_names.get(cid, f"Unknown ({cid})")
        color = colormap.get(cid, default_color)
        count = np.sum(class_id == cid)
        
        # Display the legend in the terminal
        color_name = f"RGB({color[0]}, {color[1]}, {color[2]})"
        print(f" Class {cid:2d} | {name:15s} | {count:6d} points | Color: {color_name}")
        
        mask = class_id == cid
        colors[mask] = color
        
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nStarting Interactive 3D Visualization...")
    print("Controls: \n - Left Click + Drag: Rotate \n - Scroll: Zoom \n - Shift + Left Click + Drag: Pan \n - Press 'Q' or close window to exit.")
    
    # Render
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name=f"Label Viewer - {os.path.basename(pcd_path)}",
        width=1024, 
        height=768
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Labeled PCD")
    parser.add_argument("--pcd", type=str, default="data/labeled/road_label_test/frame_00015.pcd", help="Path to PCD file")
    parser.add_argument("--map", type=str, default="data/labeled/road_label_test/class_map.json", help="Path to class_map.json")
    args = parser.parse_args()
    
    visualize_pcd(args.pcd, args.map)
