import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools

def generate_bounding_boxes_viz():
    # Load the frame
    file_path = "../data/raw/lidar_with_intensity_and_clusters/frame_00010.pcd"
    t_pcd = o3d.t.io.read_point_cloud(file_path)
    
    points = t_pcd.point.positions.numpy()
    valid_mask = np.isfinite(points).all(axis=1)
    
    valid_points = points[valid_mask]
    class_ids = t_pcd.point.class_id.numpy()[valid_mask].flatten()
    actor_ids = t_pcd.point.actor_id.numpy()[valid_mask].flatten()
    
    # Define mapping
    c_red = np.array([255, 77, 77]) / 255.0
    c_purple = np.array([153, 50, 204]) / 255.0
    c_cyan = np.array([0, 191, 255]) / 255.0
    c_green = np.array([60, 179, 113]) / 255.0
    
    colors = np.zeros((len(valid_points), 3))
    colors[np.isin(class_ids, [1, 2, 3, 4])] = c_red
    colors[np.isin(class_ids, [6])] = c_purple
    colors[np.isin(class_ids, [7, 8])] = c_cyan
    colors[np.isin(class_ids, [0, 5])] = c_green
    
    class_name_map = {
        1: "Car",
        2: "Truck",
        3: "Bicycle",
        4: "Pedestrian",
        6: "Guardrail"
    }
    
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for rendering speed and clarity (like the user's image)
    step = 5
    
    # To match the user's image perspective, let's only plot points within a certain range
    # Focus heavily in front of the vehicle
    mask_view = (valid_points[:, 0] > 0) & (valid_points[:, 0] < 30) & (valid_points[:, 1] > -10) & (valid_points[:, 1] < 10)
    
    plot_pts = valid_points[mask_view]
    plot_cols = colors[mask_view]
    
    ax.scatter(plot_pts[::step, 0], plot_pts[::step, 1], plot_pts[::step, 2], 
               c=plot_cols[::step], s=2.0, depthshade=False, edgecolors='none', marker='.')
               
    # Find unique actors in the view
    unique_actors = np.unique(actor_ids[mask_view])
    
    for actor in unique_actors:
        # Ignore unassigned or background actors (often 0 or -1 depending on dataset, here let's assume 0 is bg)
        if actor == 0:
            continue
            
        actor_mask = (actor_ids == actor) & mask_view
        if not np.any(actor_mask):
            continue
            
        actor_pts = valid_points[actor_mask]
        actor_class = int(class_ids[actor_mask][0])
        
        # Only draw bounding boxes for Dynamic (1,2,3,4) and Static Obstacles (6)
        if actor_class not in class_name_map:
            continue
            
        # Get bounding box min and max
        min_pt = np.min(actor_pts, axis=0)
        max_pt = np.max(actor_pts, axis=0)
        
        # Add a tiny margin to the box
        margin = 0.1
        min_pt -= margin
        max_pt += margin
        
        # Define the 8 corners of the box
        x = [min_pt[0], max_pt[0]]
        y = [min_pt[1], max_pt[1]]
        z = [min_pt[2], max_pt[2]]
        
        # List of lines to draw (edges of the box)
        # Using itertools to get all combinations of corners
        for s, e in itertools.combinations(np.array(list(itertools.product(x, y, z))), 2):
            # If the points share 2 coordinates, they form an edge
            if np.sum(s == e) == 2:
                # Color code the bounding box to match the class
                line_color = c_red if actor_class in [1, 2, 3, 4] else c_purple
                ax.plot3D(*zip(s, e), color='black', lw=1.5, alpha=0.8)
                ax.plot3D(*zip(s, e), color=line_color, lw=1.0)
                
        # Add Text Label at the top center of the bounding box
        mid_x = (min_pt[0] + max_pt[0]) / 2
        mid_y = (min_pt[1] + max_pt[1]) / 2
        top_z = max_pt[2] + 0.5 # Slightly above the box
        
        label_text = class_name_map[actor_class]
        label_color = c_red if actor_class in [1, 2, 3, 4] else c_purple
        
        ax.text(mid_x, mid_y, top_z, label_text, color='white',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(facecolor=label_color, edgecolor='white', boxstyle='round,pad=0.3', alpha=0.9))
                
    # Set camera perspective identical to the user's image (flat, facing forward)
    # The image is from a side/elevated perspective, azim ~ -90 (looking down X axis) or similar.
    # Actually, X is forward, Y is left. The user image shows the truck and pedestrian from the side.
    # Meaning we are looking from the side (Y axis).
    ax.view_init(elev=5, azim=-45) # Adjusted to get a good angled view of the boxes
    
    # Axis limits to frame the scene nicely
    ax.set_xlim([0, 30])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-2, 5])
    
    # Fix aspect ratio manually to prevent stretching
    ax.set_box_aspect((30, 20, 7))
    
    # Aesthetics
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    # Create custom legend
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
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, 
              facecolor='#f0f8ff', edgecolor='#99ccff', borderpad=1.5, labelspacing=1.5)
    
    plt.tight_layout()
    plt.savefig('labeled_bounding_boxes.png', bbox_inches='tight', pad_inches=0.1)
    print("Successfully saved labeled_bounding_boxes.png")

if __name__ == "__main__":
    generate_bounding_boxes_viz()
