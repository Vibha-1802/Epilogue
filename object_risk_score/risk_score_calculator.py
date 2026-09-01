import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
CSV_FILE = 'radar_detections.csv'
MAX_COAST_TIME = 0.5  # Seconds an object is kept alive without new detection
LOW_RISK_THRESHOLD = 15.0 # Risk percentage below which an object is considered low risk

def create_dummy_data():
    """Create a dummy radar_detections.csv if the MATLAB export hasn't been run."""
    print(f"Warning: {CSV_FILE} not found. Creating a demonstration dataset (Imminent Crash Scenario).")
    data = []
    # Object 1: Close Car (ID: 1)
    # Object 2: Pedestrian (ID: 2)
    # Object 3: Far Car (ID: 3)
    for t in np.arange(0, 3.0, 0.1):
        # Obj 1: starts at 10m, moves at -2.0m/s
        data.append([t, 1, 10 + (-2.0)*t, 0.0, -2.0, 0])
        
        # Obj 2: starts at 15m, stationary (Pedestrian stepping in)
        data.append([t, 2, 15, -0.28, 0.0, 0])
            
        # Obj 3: starts at 20m, moves at -5.0m/s
        data.append([t, 3, 20 + (-5.0)*t, 0.48, -5.0, 0])
        
    df = pd.DataFrame(data, columns=['Time', 'ObjectID', 'X', 'Y', 'VX', 'VY'])
    df.to_csv(CSV_FILE, index=False)

def calculate_risk(x, y, vx, vy):
    """
    Calculates a risk score from 0 to 100 based on Distance and Time-to-Collision (TTC).
    """
    distance = np.sqrt(x**2 + y**2)
    
    # Radial velocity (speed towards ego vehicle at origin)
    # Negative means approaching
    if distance > 0:
        radial_vel = (x*vx + y*vy) / distance
    else:
        radial_vel = 0
        
    # TTC calculation
    if radial_vel < -0.1: # Approaching
        ttc = distance / abs(radial_vel)
    else:
        ttc = float('inf') # Moving away or stationary
        
    # Risk Formula (Weights can be tuned)
    # Inverse distance + Inverse TTC
    dist_risk = 100.0 / (distance + 1.0)
    ttc_risk = 50.0 / (ttc + 0.1)
    
    raw_score = dist_risk + ttc_risk
    # Cap at 100%
    return min(100.0, raw_score)

def main():
    if not os.path.exists(CSV_FILE):
        create_dummy_data()
        
    df = pd.read_csv(CSV_FILE)
    
    # Active tracker state
    # Format: { ObjectID: {'last_seen': time, 'X': x, 'Y': y, 'VX': vx, 'VY': vy, 'risk': score} }
    active_objects = {}
    
    # For plotting
    history_records = []

    # Get unique sorted timestamps
    time_steps = sorted(df['Time'].unique())
    
    print("\n" + "="*70)
    print(f"RADAR RISK SCORE ANALYSIS (Coast Time = {MAX_COAST_TIME}s)")
    print("="*70)
    
    for t in time_steps:
        print(f"\n--- Frame Time: {t:.1f}s ---")
        current_dets = df[df['Time'] == t]
        
        # 1. Update detected objects
        detected_ids = set()
        for _, row in current_dets.iterrows():
            obj_id = int(row['ObjectID'])
            detected_ids.add(obj_id)
            active_objects[obj_id] = {
                'last_seen': t,
                'X': row['X'],
                'Y': row['Y'],
                'VX': row['VX'],
                'VY': row['VY']
            }
            
        # 2. Process all active objects (Detected + Coasted)
        ids_to_drop = []
        
        # Header for frame print
        print(f"{'Obj ID':<8} | {'Status':<10} | {'Dist (m)':<10} | {'TTC (s)':<10} | {'Risk Score':<10}")
        print("-" * 60)
        
        for obj_id, state in active_objects.items():
            time_since_seen = t - state['last_seen']
            
            # If the object wasn't seen this frame, but is within coast time
            is_coasting = False
            if obj_id not in detected_ids:
                if time_since_seen <= MAX_COAST_TIME:
                    # Project trajectory (Constant Velocity Model)
                    dt = time_since_seen
                    state['X'] += state['VX'] * dt
                    state['Y'] += state['VY'] * dt
                    # Update last_seen so dt calculation works correctly next frame
                    state['last_seen'] = t 
                    is_coasting = True
                else:
                    # Exceeded threshold, mark for dropping
                    ids_to_drop.append(obj_id)
                    continue
                    
            # Calculate Risk
            risk = calculate_risk(state['X'], state['Y'], state['VX'], state['VY'])
            state['risk'] = risk
            
            # Distance and TTC for display
            dist = np.sqrt(state['X']**2 + state['Y']**2)
            rad_vel = (state['X']*state['VX'] + state['Y']*state['VY']) / dist if dist > 0 else 0
            ttc = dist / abs(rad_vel) if rad_vel < -0.1 else float('inf')
            
            status_str = "Coasting" if is_coasting else "Active"
            ttc_str = f"{ttc:.1f}" if ttc != float('inf') else "N/A"
            
            print(f"{obj_id:<8} | {status_str:<10} | {dist:<10.1f} | {ttc_str:<10} | {risk:>6.1f}%")
            
            # Save for plotting
            history_records.append({'Time': t, 'ObjectID': obj_id, 'Risk': risk})
            
        # 3. Drop stale objects
        for obj_id in ids_to_drop:
            print(f">>> Object {obj_id} dropped (Not seen for > {MAX_COAST_TIME}s)")
            del active_objects[obj_id]
            
    # --- VISUALIZATION ---
    if len(history_records) > 0:
        history_df = pd.DataFrame(history_records)
        plt.figure(figsize=(10, 6))
        
        for obj_id in history_df['ObjectID'].unique():
            obj_data = history_df[history_df['ObjectID'] == obj_id]
            plt.plot(obj_data['Time'], obj_data['Risk'], marker='o', label=f'Object {obj_id}')
            
        plt.axhline(y=LOW_RISK_THRESHOLD, color='r', linestyle='--', label='Low Risk Threshold')
        plt.title('Object Risk Scores Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Risk Score (%)')
        plt.grid(True)
        plt.legend()
        
        plot_path = 'risk_score_plot.png'
        plt.savefig(plot_path)
        print(f"\nSaved visualization plot to {plot_path}")

if __name__ == "__main__":
    main()
