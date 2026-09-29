import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def generate_kdtree_viz():
    # Load authentic project data
    file_path = "../data/raw/lidar_with_intensity_and_clusters/frame_00007.pcd"
    t_pcd = o3d.t.io.read_point_cloud(file_path)
    
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    
    # Let's crop a small interesting area (e.g., around a vehicle or object)
    # The car is at origin (0,0,0), let's pick a patch slightly in front (e.g. X between 5 and 10, Y between -2 and 2)
    crop_mask = (valid_points[:, 0] > 5) & (valid_points[:, 0] < 10) & \
                (valid_points[:, 1] > -3) & (valid_points[:, 1] < 3) & \
                (valid_points[:, 2] > -2) & (valid_points[:, 2] < 2)
    
    patch = valid_points[crop_mask]
    
    # Subsample patch so it looks like a clean point cloud neighborhood
    np.random.seed(42)
    if len(patch) > 1000:
        idx = np.random.choice(len(patch), 1000, replace=False)
        patch = patch[idx]
        
    # Build cKDTree for authentic nearest-neighbor querying
    tree = cKDTree(patch)
    
    # Pick a query point (e.g. somewhere near the center of the patch)
    query_point = np.array([7.5, 0.0, 0.0])
    
    # Find the nearest point in the patch to act as our actual query point
    _, center_idx = tree.query(query_point, k=1)
    query_pt = patch[center_idx]
    
    # Query 16 nearest neighbors (we ask for 17 because the point itself counts as 1)
    k_neighbors = 16
    distances, indices = tree.query(query_pt, k=k_neighbors + 1)
    
    # The first index is the point itself, skip it for the neighbor connections
    neighbor_pts = patch[indices[1:]]
    
    # Setup plotting
    fig = plt.figure(figsize=(12, 10), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot background points
    ax.scatter(patch[:, 0], patch[:, 1], patch[:, 2], 
               c='lightgray', s=10, alpha=0.5, label='LiDAR Point Cloud')
               
    # Plot neighbor points
    ax.scatter(neighbor_pts[:, 0], neighbor_pts[:, 1], neighbor_pts[:, 2], 
               c='deepskyblue', s=60, alpha=1.0, edgecolor='white', label=f'{k_neighbors} Nearest Neighbors')
               
    # Plot query point
    ax.scatter(query_pt[0], query_pt[1], query_pt[2], 
               c='crimson', s=120, marker='*', edgecolor='black', label='Query Point (Centroid)')
               
    # Draw KD-Tree graph connections
    for n_pt in neighbor_pts:
        ax.plot([query_pt[0], n_pt[0]], 
                [query_pt[1], n_pt[1]], 
                [query_pt[2], n_pt[2]], 
                color='crimson', alpha=0.6, linewidth=1.5, linestyle='--')
                
    # Formatting
    ax.view_init(elev=20, azim=45)
    ax.set_title("cKDTree Point Cloud Neighborhood (k=16)", fontsize=18, pad=20, fontweight='bold')
    
    # Minimalist axes for clean look
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Remove tick labels for a clean schematic look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.legend(loc='lower left', fontsize=12, frameon=True, facecolor='white', edgecolor='lightgray', borderpad=1)
    
    plt.tight_layout()
    plt.savefig('kdtree_16nn_visual.png', bbox_inches='tight', dpi=300)
    print("Saved kdtree_16nn_visual.png")

if __name__ == "__main__":
    generate_kdtree_viz()
