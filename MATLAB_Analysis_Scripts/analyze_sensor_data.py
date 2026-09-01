import scipy.io
import pandas as pd
import sys
import numpy as np

def main():
    print("🚀 Analyzing sensor_data.mat...")
    try:
        # We use squeeze_me=True to remove unnecessary 1D array wrappers
        data = scipy.io.loadmat("sensor_data.mat", squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        sys.exit(1)
        
    radar = data["radar_matrix"]
    
    # Depending on how the file was exported, Detections might be nested or direct
    if hasattr(radar, 'Detections'):
        detections = radar.Detections
    else:
        # If it's a direct array of structures
        detections = radar
    
    print(f"Found {len(detections)} radar detections. Extracting...")
    
    parsed_data = []
    
    for i, det in enumerate(detections):
        # We safely try to get the attributes, defaulting to None if they don't exist
        time_val = getattr(det, 'Time', None)
        meas = getattr(det, 'Measurement', None)
        sensor_idx = getattr(det, 'SensorIndex', None)
        class_id = getattr(det, 'ObjectClassID', None)
        
        # Handle Measurement array conversion if possible
        meas_str = str(meas)
        if isinstance(meas, np.ndarray):
            # If it's a numeric array, flatten it to a list for CSV
            if np.issubdtype(meas.dtype, np.number):
                meas_str = str(meas.flatten().tolist())
            else:
                meas_str = "MCOS_Nested_Object"
                
        parsed_data.append({
            "Detection_ID": i,
            "Time": time_val,
            "SensorIndex": sensor_idx,
            "ObjectClassID": class_id,
            "Measurement": meas_str
        })
        
    df = pd.DataFrame(parsed_data)
    print("\n✅ Successfully extracted Detections to DataFrame:")
    print(df.head())
    
    # Save to CSV so it's readable
    df.to_csv("radar_detections.csv", index=False)
    print("\n💾 Saved clean data to 'radar_detections.csv'")

if __name__ == "__main__":
    main()
