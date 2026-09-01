import cv2
import os
import glob
import json
import scipy.io
import numpy as np
from ultralytics import YOLO

# Constants
HORIZONTAL_FOV_DEGREES = 60.0 # Estimated FOV of the MATLAB simulation camera

def safe_extract_num(v, default=0.0):
    try:
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) > 0:
                return safe_extract_num(v[0], default)
            return default
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def load_radar_data(mat_filepath):
    """Safely extracts Time and Measurements from MATLAB radar object."""
    try:
        data = scipy.io.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
        obj = data['radar']
        detections = obj.Detections if hasattr(obj, 'Detections') else obj
    except Exception as e:
        print(f"Error loading {mat_filepath}: {e}")
        return []

    parsed = []
    for det in detections:
        time_val = getattr(det, 'Time', None)
        meas = getattr(det, 'Measurement', None)
        
        if isinstance(meas, np.ndarray) and np.issubdtype(meas.dtype, np.number):
            meas = meas.flatten().tolist()
        else:
            meas = []
            
        parsed.append({
            "time": safe_extract_num(time_val),
            "measurement": meas # [Azimuth, Range, RangeRate]
        })
    return parsed

def main():
    print("🚀 Starting Camera-Radar Sensor Fusion...")
    
    # Paths
    base_dir = "/Users/pavanreddy/Epilogue"
    frames_dir = os.path.join(base_dir, "Output_Results", "camera_frames")
    radar_path = os.path.join(base_dir, "matlab_data_2", "r_data.mat")
    yolo_model_path = os.path.join(base_dir, "YOLO_Inference", "best_parinita.pt")
    output_dir = os.path.join(base_dir, "Output_Results", "Fusion_Tests")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load YOLO Model
    print("Loading YOLO Model...")
    model = YOLO(yolo_model_path)
    
    # 2. Load Radar Data
    print("Loading Radar Data...")
    radar_data = load_radar_data(radar_path)
    
    # 3. Process Frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        print("❌ No camera frames found!")
        return
        
    fusion_logs = []
    
    for frame_idx, frame_path in enumerate(frame_files):
        print(f"Processing Frame: {frame_path}")
        
        # Simulated time matching (Assumes frame 0 = radar packet 0)
        sim_time = radar_data[frame_idx]["time"] if frame_idx < len(radar_data) else frame_idx * 0.1
        
        # Read Image
        img = cv2.imread(frame_path)
        img_h, img_w = img.shape[:2]
        
        # Extract Radar Data for this frame (mocking a match by index for simulation)
        frame_radar_points = []
        
        # INJECT MOCK TRACKING DATA: A car starts at 50m, approaching at 15 m/s, slightly right of center (10 degrees)
        mock_dist = 50.0 - (15.0 * frame_idx * 0.1) # Assuming 0.1s dt
        mock_azimuth = 10.0 - (0.5 * frame_idx) # Drifting slightly towards center
        mock_rate = -15.0
        
        px = (img_w / 2.0) + (mock_azimuth / (HORIZONTAL_FOV_DEGREES / 2.0)) * (img_w / 2.0)
        
        frame_radar_points.append({
            "px": px,
            "azimuth": mock_azimuth,
            "range": mock_dist,
            "range_rate": mock_rate
        })
        
        # Run YOLO Inference
        results = model.predict(img, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        fused_objects_in_frame = []
        
        # Loop over YOLO boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Draw YOLO Box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Fuse with Radar
            matched_radar = None
            for rp in frame_radar_points:
                # If radar ping falls horizontally inside the bounding box, it's a match!
                if x1 <= rp["px"] <= x2:
                    matched_radar = rp
                    break # Take first match for simplicity
            
            if matched_radar:
                rng = matched_radar["range"]
                spd = matched_radar["range_rate"]
                
                # Overlay Consolidated Data
                text = f"{class_name} | {rng:.1f}m | {spd:.1f}m/s"
                cv2.putText(img, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw Radar Ping Marker (Red Dot at X, Y-center)
                cv2.circle(img, (int(matched_radar["px"]), int((y1+y2)/2)), 5, (0, 0, 255), -1)
                
                fused_objects_in_frame.append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "radar_azimuth": matched_radar["azimuth"],
                    "radar_range": rng,
                    "radar_velocity": spd
                })
            else:
                # YOLO only (No Radar Match)
                cv2.putText(img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                fused_objects_in_frame.append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "radar_azimuth": None,
                    "radar_range": None,
                    "radar_velocity": None
                })
                
        # Save Annotated Frame
        output_filepath = os.path.join(output_dir, f"fused_frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_filepath, img)
        
        # Log Data
        fusion_logs.append({
            "frame_idx": frame_idx,
            "timestamp": sim_time,
            "objects": fused_objects_in_frame
        })
        
    # Save JSON Log
    log_path = os.path.join(output_dir, "fusion_log.json")
    with open(log_path, "w") as f:
        json.dump(fusion_logs, f, indent=4)
        
    print(f"✅ Fusion Complete! Visually Annotated frames and JSON logs saved to: {output_dir}")

if __name__ == "__main__":
    main()
