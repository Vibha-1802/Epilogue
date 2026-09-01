import scipy.io
import pandas as pd
import numpy as np
import os
import sys

def parse_detections(mat_filepath, sensor_name):
    print(f"[{sensor_name}] Parsing {mat_filepath}...")
    try:
        data = scipy.io.loadmat(mat_filepath, squeeze_me=True, struct_as_record=False)
        obj = data[sensor_name]
    except Exception as e:
        print(f"  ❌ Error loading {mat_filepath}: {e}")
        return None
    
    if hasattr(obj, 'Detections'):
        detections = obj.Detections
    else:
        detections = obj
        
    print(f"  Found {len(detections)} {sensor_name} detections.")
    
    parsed = []
    for i, det in enumerate(detections):
        time_val = getattr(det, 'Time', None)
        meas = getattr(det, 'Measurement', None)
        sensor_idx = getattr(det, 'SensorIndex', None)
        class_id = getattr(det, 'ObjectClassID', None)
        
        meas_str = str(meas)
        if isinstance(meas, np.ndarray):
            if np.issubdtype(meas.dtype, np.number):
                meas_str = str(meas.flatten().tolist())
            else:
                meas_str = "MCOS_Nested_Object"
                
        parsed.append({
            "Detection_ID": i,
            "Time": time_val,
            "SensorIndex": sensor_idx,
            "ObjectClassID": class_id,
            "Measurement": meas_str
        })
        
    df = pd.DataFrame(parsed)
    return df

def main():
    print("🚀 Extracting matlab_data_3...")
    
    os.makedirs("Results", exist_ok=True)
    
    # 1. Parse Radar
    if os.path.exists("r_data.mat"):
        radar_df = parse_detections("r_data.mat", "radar")
        if radar_df is not None:
            radar_df.to_csv("Results/radar_detections.csv", index=False)
            print("  💾 Saved to Results/radar_detections.csv")
    
    # 2. Check Camera
    if not os.path.exists("c_data.mat"):
        print("[camera] ⚠️ c_data.mat is MISSING. Skipping camera extraction.")
        
    # 3. Lidar Summary
    lidar_dir = "Lidar_PCD_Data"
    if os.path.exists(lidar_dir):
        files = [f for f in os.listdir(lidar_dir) if f.endswith('.pcd')]
        print(f"[lidar] Found {len(files)} .pcd files in {lidar_dir}.")

    print("\n✅ Extraction complete! Check the 'Results' folder.")

if __name__ == "__main__":
    main()
