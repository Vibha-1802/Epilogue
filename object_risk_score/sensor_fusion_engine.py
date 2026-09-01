import json
import math
import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import trajectory_planner as tp
import matplotlib.path as mplPath
import matplotlib.pyplot as plt

# ==========================================
# SENSOR FUSION ENGINE
# Mapping Radar Blips to Camera Semantic Polygons
# ==========================================

# 1. Camera Configuration (Heuristic)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FOV_DEG = 90.0 # Assumed Field of View of the camera
HORIZON_Y = 360 # Roughly the middle of the image

def polar_to_cartesian(range_val, azimuth_deg, range_rate):
    """
    Converts Radar Polar coordinates to Cartesian X, Y.
    Assuming Azimuth 0 is straight ahead (Longitudinal X).
    """
    azimuth_rad = math.radians(azimuth_deg)
    
    # Longitudinal distance (forward)
    x = range_val * math.cos(azimuth_rad)
    # Lateral distance (left/right)
    y = range_val * math.sin(azimuth_rad)
    
    vx = range_rate * math.cos(azimuth_rad)
    vy = range_rate * math.sin(azimuth_rad)
    
    return x, y, vx, vy

def heuristic_projection(range_val, azimuth_deg):
    """
    Since we don't have exact Extrinsic/Intrinsic matrices,
    we use a heuristic to map Radar (Range, Azimuth) to Camera Pixels (U, V).
    """
    # 1. Horizontal mapping (Azimuth -> U pixel)
    # Azimuth of -45 deg -> pixel 0, Azimuth of +45 deg -> pixel 1280
    u = (IMAGE_WIDTH / 2.0) + (azimuth_deg / (CAMERA_FOV_DEG / 2.0)) * (IMAGE_WIDTH / 2.0)
    
    # 2. Vertical mapping (Range -> V pixel)
    # Objects further away (Range > 100m) approach the horizon line.
    # Objects close up (Range = 5m) drop to the bottom of the frame.
    if range_val <= 0.1: range_val = 0.1
    # Simple perspective drop-off
    drop = 500.0 / range_val 
    v = HORIZON_Y + drop
    
    # Bound pixels to image size
    u = max(0, min(IMAGE_WIDTH, u))
    v = max(0, min(IMAGE_HEIGHT, v))
    
    return u, v

def parse_camera_polygons():
    """
    Returns mock Camera Polygons (bounding boxes) for 5 generated objects in a crowded scene.
    """
    # Class 0: Car (Dead center, Range 15 -> U=640)
    car_pts = [[600.0, 370.0], [680.0, 370.0], [680.0, 400.0], [600.0, 400.0]]
    
    # Class 6: Pedestrian (Azimuth 14.0 -> U=839)
    ped_pts = [[810.0, 390.0], [870.0, 390.0], [870.0, 430.0], [810.0, 430.0]]
    
    # Class 5: Bicycle (Azimuth -11.7 -> U=473)
    bike_pts = [[440.0, 380.0], [500.0, 380.0], [500.0, 420.0], [440.0, 420.0]]
    
    # Class 20: Pothole (Azimuth 2.8 -> U=679)
    pothole_pts = [[650.0, 380.0], [710.0, 380.0], [710.0, 410.0], [650.0, 410.0]]
    
    # Class 2: Truck (Dead center further up, Range 25 -> U=640)
    truck_pts = [[580.0, 360.0], [700.0, 360.0], [700.0, 390.0], [580.0, 390.0]]
    
    polygons = {
        0: mplPath.Path(car_pts),
        6: mplPath.Path(ped_pts),
        5: mplPath.Path(bike_pts),
        20: mplPath.Path(pothole_pts),
        2: mplPath.Path(truck_pts)
    }
    
    return polygons

def assign_semantic_classes(radar_blips, polygons):
    """
    Point-in-Polygon logic:
    Finds which camera polygon the radar pixel falls into.
    """
    for blip in radar_blips:
        point = (blip['u'], blip['v'])
        assigned = False
        for class_id, poly in polygons.items():
            if poly.contains_point(point):
                blip['class_id'] = class_id
                assigned = True
                break
        
        # If it doesn't fall in any polygon, assume it's just 'unknown' (e.g. background)
        if not assigned:
            blip['class_id'] = -1 
            
    return radar_blips

def get_fused_objects():
    """
    Simulates the sensor fusion pipeline for a single frame, returning the initial world state.
    """
    raw_radar_data = [
        {'id': 1, 'azimuth': 0.0, 'range': 15.0, 'range_rate': -2.0}, # Center (Car)
        {'id': 2, 'azimuth': 14.0, 'range': 10.3, 'range_rate': 0.0}, # Far Left (Pedestrian)
        {'id': 3, 'azimuth': -11.7, 'range': 12.25, 'range_rate': 0.0}, # Far Right (Bicycle)
        {'id': 4, 'azimuth': 2.8, 'range': 20.0, 'range_rate': 0.0}, # Left-Center (Pothole)
        {'id': 5, 'azimuth': 0.0, 'range': 25.0, 'range_rate': -5.0}, # Center further up (Truck)
    ]
    
    radar_blips = []
    for data in raw_radar_data:
        x, y, vx, vy = polar_to_cartesian(data['range'], data['azimuth'], data['range_rate'])
        u, v = heuristic_projection(data['range'], data['azimuth'])
        radar_blips.append({
            'obj_id': data['id'], 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'u': u, 'v': v, 'class_id': None
        })
        
    polygons = parse_camera_polygons()
    fused_blips = assign_semantic_classes(radar_blips, polygons)
    
    world_objects = []
    for blip in fused_blips:
        if blip['class_id'] != -1:
            obj = tp.WorldObject(
                obj_id=blip['obj_id'], class_id=blip['class_id'], 
                x=blip['x'], y=blip['y'], vx=blip['vx'], vy=blip['vy'], risk_score=50
            )
            world_objects.append(obj)
            
    return world_objects

def main():
    print("Initializing Sensor Fusion Node...")
    world_objects = get_fused_objects()

    # 5. Output ready for Trajectory Planner
    print("\nPassing fused objects directly to trajectory_planner.py...")
    
    # Run trajectory planner with the dynamically fused objects
    time_horizon = 3.0
    ego_speed = 10.0
    predicted_obs = tp.predict_obstacle_trajectories(world_objects, time_horizon)
    candidates = tp.generate_candidate_trajectories(ego_speed, time_horizon)
    
    best_traj = None
    min_cost = float('inf')
    
    print("\nEvaluating Trajectories:")
    for tr in candidates:
        cost = tp.evaluate_trajectory_cost(tr, predicted_obs, world_objects)
        print(f"Path Offset {tr['offset']:>4.1f}m -> Cost: {cost:.2f}")
        if cost < min_cost:
            min_cost = cost
            best_traj = tr
            
    print(f"\n>> Selected Best Path: Offset {best_traj['offset']}m")
    
    # Export Controls for Simulink
    aeb_triggered = min_cost > 1000000
    tp.export_simulink_controls(best_traj, aeb_triggered, time_horizon)
    
    tp.plot_scene(world_objects, best_traj, candidates, predicted_obs)

if __name__ == "__main__":
    main()
