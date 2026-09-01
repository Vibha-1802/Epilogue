import json
import os
import math

def calculate_ttc(range_val, range_rate):
    """Calculates Time-To-Collision (TTC)."""
    if range_rate >= -0.1: # Not approaching fast enough, or moving away
        return float('inf')
    return abs(range_val / range_rate)

def predict_trajectory():
    print("🚀 Starting Trajectory Prediction & TTC Calculation...")
    
    base_dir = "/Users/pavanreddy/Epilogue"
    fusion_log_path = os.path.join(base_dir, "Output_Results", "Fusion_Tests", "fusion_log.json")
    output_path = os.path.join(base_dir, "Output_Results", "Fusion_Tests", "trajectory_predictions.json")
    
    if not os.path.exists(fusion_log_path):
        print(f"❌ Fusion log not found at {fusion_log_path}")
        return
        
    with open(fusion_log_path, "r") as f:
        fusion_data = json.load(f)
        
    # We will track objects across frames based on their Class (simplification for this simulation)
    # In a real system, you'd use a Kalman Filter tracking ID.
    tracking_history = {}
    
    dt = 0.1 # Delta time between frames in seconds
    prediction_time = 3.0 # Seconds into the future
    
    predictions = []
    
    for frame in fusion_data:
        frame_idx = frame["frame_idx"]
        sim_time = frame["timestamp"]
        
        frame_predictions = {
            "frame_idx": frame_idx,
            "timestamp": sim_time,
            "tracked_objects": []
        }
        
        for obj in frame["objects"]:
            cls = obj["class"]
            r_range = obj.get("radar_range")
            r_azimuth = obj.get("radar_azimuth")
            r_velocity = obj.get("radar_velocity")
            
            # If the object doesn't have radar data, we can't accurately predict Cartesian trajectory
            if r_range is None or r_azimuth is None:
                continue
                
            # Convert Spherical to Cartesian (X = Forward, Y = Left/Right)
            azimuth_rad = math.radians(r_azimuth)
            x_current = r_range * math.cos(azimuth_rad)
            y_current = r_range * math.sin(azimuth_rad)
            
            # Velocity Tracking
            vx = 0.0
            vy = 0.0
            
            if cls in tracking_history:
                # Calculate velocity based on displacement from previous frame
                prev_x = tracking_history[cls]["x"]
                prev_y = tracking_history[cls]["y"]
                vx = (x_current - prev_x) / dt
                vy = (y_current - prev_y) / dt
            else:
                # If it's the first frame, use the radar's radial velocity as an approximation for Vx
                vx = r_velocity * math.cos(azimuth_rad)
                vy = r_velocity * math.sin(azimuth_rad)
                
            # Update history
            tracking_history[cls] = {"x": x_current, "y": y_current}
            
            # Predict Future Position (Constant Velocity Model)
            x_future = x_current + (vx * prediction_time)
            y_future = y_current + (vy * prediction_time)
            
            # Calculate TTC
            ttc = calculate_ttc(r_range, r_velocity)
            
            frame_predictions["tracked_objects"].append({
                "class": cls,
                "current_position": {"x": round(x_current, 2), "y": round(y_current, 2)},
                "velocity": {"vx": round(vx, 2), "vy": round(vy, 2)},
                "future_position_3s": {"x": round(x_future, 2), "y": round(y_future, 2)},
                "time_to_collision_sec": round(ttc, 2) if ttc != float('inf') else "Safe"
            })
            
            print(f"[Frame {frame_idx}] Tracked {cls}: Dist={r_range:.1f}m | Vel=[{vx:.1f}, {vy:.1f}]m/s | TTC={ttc:.1f}s")
            
        predictions.append(frame_predictions)
        
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
        
    print(f"\n✅ Trajectory Predictions successfully saved to {output_path}")

if __name__ == "__main__":
    predict_trajectory()
