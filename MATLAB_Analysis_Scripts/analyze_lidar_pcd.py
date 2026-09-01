import os
import sys

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("❌ Error: You need the 'open3d' and 'numpy' libraries for LiDAR processing.")
    print("Please install them by running: pip install open3d numpy")
    sys.exit(1)

def main():
    print("🚀 LiDAR Point Cloud (.pcd) Analysis Script")
    print("---------------------------------------------")
    
    # 1. Ask the user for the .pcd file
    pcd_file = input("Enter the path to your exported .pcd file: ").strip().strip("'\"")
    
    if not os.path.exists(pcd_file):
        print(f"❌ Error: File '{pcd_file}' not found.")
        print("Remember: You must export your LiDAR data from MATLAB as a .pcd file!")
        sys.exit(1)
        
    print(f"\nLoading '{pcd_file}'...")
    
    # 2. Load the Point Cloud using Open3D
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # 3. Print basic statistics
    points = np.asarray(pcd.points)
    print("\n✅ Successfully loaded Point Cloud!")
    print(f"Total Points: {len(points)}")
    if len(points) > 0:
        print(f"X bounds: {points[:, 0].min():.2f} to {points[:, 0].max():.2f} meters")
        print(f"Y bounds: {points[:, 1].min():.2f} to {points[:, 1].max():.2f} meters")
        print(f"Z bounds: {points[:, 2].min():.2f} to {points[:, 2].max():.2f} meters")
    
    # 4. Demonstrate Ground Plane Removal (RANSAC)
    print("\n--- Processing Step 1: Removing Ground Plane ---")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    print(f"Removed {len(inliers)} ground points.")
    print(f"Remaining {len(outlier_cloud.points)} object points to cluster.")
    
    # 5. DBSCAN Clustering
    print("\n--- Processing Step 2: DBSCAN Clustering ---")
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))
    
    max_label = labels.max()
    print(f"Detected {max_label + 1} distinct objects (clusters) in the scene!")
    
    # 6. Visualization
    print("\nOpening 3D Visualizer... (Close the window to exit)")
    
    # Color the ground grey, and give the objects random colors
    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0 # Noise points in black
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="LiDAR DBSCAN Clustering")

if __name__ == "__main__":
    main()
