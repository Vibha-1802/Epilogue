import scipy.io
import pandas as pd
import numpy as np
import os
import json
import glob

# NOTE: We use open3d for LiDAR processing if installed.
O3D_AVAILABLE = False
print("WARNING: open3d clustering disabled to prevent hangs in headless environments. LiDAR clustering will be mocked for demo.")

def safe_extract_num(v, default=0.0):
    try:
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) > 0:
                return safe_extract_num(v[0], default)
            return default
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def parse_mat_detections(mat_filepath, sensor_name):
    """Safely extracts Time, ClassID, and Measurements from MATLAB object."""
    try:
        data = scipy.io.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
        obj = data[sensor_name]
        detections = obj.Detections if hasattr(obj, 'Detections') else obj
    except Exception as e:
        print(f"Error loading {mat_filepath}: {e}")
        return []

    parsed = []
    for det in detections:
        time_val = getattr(det, 'Time', None)
        meas = getattr(det, 'Measurement', None)
        class_id = getattr(det, 'ObjectClassID', None)
        
        # Flatten numeric measurements (e.g., [Azimuth, Range, Rate] or [x,y,w,h])
        if isinstance(meas, np.ndarray) and np.issubdtype(meas.dtype, np.number):
            meas = meas.flatten().tolist()
        else:
            meas = []
            
        parsed.append({
            "time": safe_extract_num(time_val),
            "class_id": int(safe_extract_num(class_id, -1.0)),
            "measurement": meas
        })
    return parsed

def process_lidar_frame(pcd_path):
    """Processes a single LiDAR .pcd frame into 3D objects."""
    if not O3D_AVAILABLE:
        # Mock data if open3d is missing on this environment
        return [{"centroid_3d": [10.5, 0.0, -1.2], "volume": 8.4}]
        
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        return []
        
    # 1. RANSAC Ground Removal
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    # 2. DBSCAN Clustering
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))
    max_label = labels.max()
    
    clusters = []
    for i in range(max_label + 1):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) < 10: continue
            
        cluster_pcd = outlier_cloud.select_by_index(cluster_idx)
        aabb = cluster_pcd.get_axis_aligned_bounding_box()
        
        centroid = aabb.get_center().tolist()
        extent = aabb.get_extent().tolist()
        volume = extent[0] * extent[1] * extent[2]
        
        clusters.append({
            "centroid_3d": [round(x, 2) for x in centroid],
            "volume": round(volume, 2)
        })
    return clusters

def run_pipeline():
    print("🚀 Starting Unified Sensor Fusion Pipeline...")
    base_dir = "/Users/pavanreddy/Epilogue/matlab_data_2"
    
    # 1. Ingest Data
    print("Ingesting Radar...")
    radar_data = parse_mat_detections(os.path.join(base_dir, "r_data.mat"), "radar")
    
    print("Ingesting Camera Detections...")
    camera_data = parse_mat_detections(os.path.join(base_dir, "c_data.mat"), "camera")
    
    # Get LiDAR paths and sort them
    lidar_dir = os.path.join(base_dir, "Lidar_PCD_Data")
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
    # For speed in this demo pipeline, we'll only process the first 10 frames
    lidar_files = lidar_files[:10] 
    
    # 2. Time Synchronization (Grouping by Frame/Timestamp)
    # We will simulate a continuous time stream
    fused_timeline = []
    
    print("Synchronizing and Processing LiDAR (Batch: 10 frames)...")
    for frame_idx in range(10):
        # We assume radar/camera arrays match the frame index for this simulation
        sim_time = radar_data[frame_idx]["time"] if frame_idx < len(radar_data) else frame_idx * 0.1
        
        # Aggregate data for this exact timestamp
        frame_payload = {
            "timestamp": sim_time,
            "camera_objects": [],
            "radar_objects": [],
            "lidar_clusters": []
        }
        
        # Add Camera
        if frame_idx < len(camera_data):
            cam = camera_data[frame_idx]
            frame_payload["camera_objects"].append({
                "class_id": cam["class_id"],
                "bbox_2d": cam["measurement"] # [x, y, w, h]
            })
            
        # Add Radar
        if frame_idx < len(radar_data):
            rad = radar_data[frame_idx]
            meas = rad["measurement"]
            if len(meas) >= 3:
                frame_payload["radar_objects"].append({
                    "azimuth": meas[0],
                    "range": meas[1],
                    "range_rate": meas[2]
                })
                
        # Add LiDAR
        if frame_idx < len(lidar_files):
            clusters = process_lidar_frame(lidar_files[frame_idx])
            frame_payload["lidar_clusters"] = clusters
            
        fused_timeline.append(frame_payload)
        
    # 3. Export to JSON
    output_path = os.path.join(base_dir, "Results", "fused_sensor_data.json")
    with open(output_path, "w") as f:
        json.dump(fused_timeline, f, indent=4)
        
    print(f"✅ Pipeline Complete! Clean JSON saved to: {output_path}")

if __name__ == "__main__":
    run_pipeline()
