import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

def generate_foveated_grid_viz():
    # Load authentic project data from ryan_final
    file_path = "../lidar_dataset/Exported_PCD_Files_ryan_Final/frame_0013.pcd"
    t_pcd = o3d.t.io.read_point_cloud(file_path)
    
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    
    # We only care about X, Y for Top-Down (BEV) view
    x = valid_points[:, 0]
    y = valid_points[:, 1]
    
    # Calculate Euclidean distance (sqrt(x^2 + y^2)) for circular rings
    dist = np.sqrt(x**2 + y**2)
    
    # Radii from config.py: 10.0, 20.0, 40.0, 100.0
    r1, r2, r3, r4 = 10.0, 20.0, 40.0, 100.0
    
    # Create masks for each ring
    mask_ring0 = dist < r1
    mask_ring1 = (dist >= r1) & (dist < r2)
    mask_ring2 = (dist >= r2) & (dist < r3)
    mask_ring3 = (dist >= r3) & (dist < r4)
    
    # Colors matching the requested theme
    c_ring0 = '#ff4d4d' # Red
    c_ring1 = '#ffcc00' # Yellow
    c_ring2 = '#4CAF50' # Green
    c_ring3 = '#3498db' # Blue
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    
    # Plot points
    # Subsample heavily for rendering speed and to preserve point cloud aesthetic (preventing solid blobs)
    step = 5
    ax.scatter(y[mask_ring3][::step], x[mask_ring3][::step], c=c_ring3, s=0.2, alpha=0.5, edgecolors='none', marker='.')
    ax.scatter(y[mask_ring2][::step], x[mask_ring2][::step], c=c_ring2, s=0.5, alpha=0.6, edgecolors='none', marker='.')
    ax.scatter(y[mask_ring1][::step], x[mask_ring1][::step], c=c_ring1, s=1.0, alpha=0.8, edgecolors='none', marker='.')
    ax.scatter(y[mask_ring0][::step], x[mask_ring0][::step], c=c_ring0, s=2.0, alpha=1.0, edgecolors='none', marker='.')
    
    # Draw Euclidean Rings (Circles)
    for r, c, ls in [(r1, c_ring0, '-'), (r2, c_ring1, '-'), (r3, c_ring2, '-'), (r4, c_ring3, '--')]:
        circ = Circle((0, 0), r, linewidth=2, edgecolor=c, facecolor='none', linestyle=ls, alpha=0.9)
        ax.add_patch(circ)
        
    # Draw Ego Vehicle (simplified as a black rectangle at origin)
    ego = Rectangle((-1.0, -2.5), 2.0, 5.0, linewidth=2, edgecolor='white', facecolor='black', zorder=10)
    ax.add_patch(ego)
    
    # Formatting the plot
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect('equal')
    
    # Minimalist aesthetics
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_title("Foveated Concentric Grid (Ryan Intersection Data)", fontsize=22, fontweight='bold', pad=20)
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Inner Ring (0-10m, 5cm res)', markerfacecolor=c_ring0, markersize=14),
        Line2D([0], [0], marker='s', color='w', label='Middle Ring (10-20m, 10cm res)', markerfacecolor=c_ring1, markersize=14),
        Line2D([0], [0], marker='s', color='w', label='Outer Ring (20-40m, 20cm res)', markerfacecolor=c_ring2, markersize=14),
        Line2D([0], [0], marker='s', color='w', label='Far Ring (40-100m, 40cm res)', markerfacecolor=c_ring3, markersize=14),
        Line2D([0], [0], marker='s', color='w', label='Ego Vehicle', markerfacecolor='black', markersize=14)
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=True, 
              facecolor='white', edgecolor='lightgray', borderpad=1.2, labelspacing=1.2)
    
    plt.tight_layout()
    plt.savefig('foveated_grid_ryan_intersection.png', bbox_inches='tight', dpi=300)
    print("Saved foveated_grid_ryan_intersection.png")

if __name__ == "__main__":
    generate_foveated_grid_viz()
