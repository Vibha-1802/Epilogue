import os
import sys
import cv2
import math
import numpy as np
import scipy.io as sio
from ultralytics import YOLO
import logging
import time

# Suppress YOLO logs
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Import local trajectory and risk modules
import risk_score_calculator
import trajectory_planner

def main():
    print("Initializing Real-Time Controller...")
    
    # 1. Setup Models & Directories
    model_path = '/Users/pavanreddy/Epilogue/vehicle_behaviour_pipeline/best_parinita.pt'
    model = YOLO(model_path)
    
    input_video = '/Users/pavanreddy/Epilogue/YOLO_Inference/matlab_data_4/cam_video.mp4'
    cap = cv2.VideoCapture(input_video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    dt = 1.0 / fps
    
    output_dir = '/Users/pavanreddy/Epilogue/vehicle_behaviour_pipeline/realtime_controls'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean previous .mat files in directory for a fresh run
    for f in os.listdir(output_dir):
        if f.endswith('.mat'):
            os.remove(os.path.join(output_dir, f))
            
    print(f"Output directory initialized at {output_dir}")

    # 2. Dynamic Controller State
    v_current = 10.0  # Initial cruise speed (m/s)
    v_min = 2.0       # Minimum coasting speed (m/s) to ensure we NEVER stop completely
    v_target = 10.0   # Target cruising speed
    wheelbase = 2.5
    time_horizon = 3.0

    frame_idx = 0
    total_latency = 0.0
    
    print("Starting real-time vehicle simulation...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.time()
        
        # --- PERCEPTION (YOLO) ---
        results = model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        world_objects = []
        
        # --- SENSOR FUSION MOCK & RISK ---
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            
            w = x2 - x1
            mock_dist = max(3.0, 1500.0 / (w + 1.0)) 
            img_center_x = frame.shape[1] / 2
            obj_center_x = (x1 + x2) / 2
            mock_azimuth = (obj_center_x - img_center_x) / img_center_x * 30.0 
            mock_rate = -5.0 
            
            azimuth_rad = math.radians(mock_azimuth)
            x_current = mock_dist * math.cos(azimuth_rad)
            y_current = mock_dist * math.sin(azimuth_rad)
            vx = mock_rate * math.cos(azimuth_rad)
            vy = mock_rate * math.sin(azimuth_rad)
            
            risk_score = risk_score_calculator.calculate_risk(x_current, y_current, vx, vy)
            
            obj = trajectory_planner.WorldObject(
                obj_id=i, class_id=cls_id, 
                x=x_current, y=y_current, 
                vx=vx, vy=vy, risk_score=risk_score
            )
            world_objects.append(obj)
            
        # --- TRAJECTORY PLANNING ---
        if world_objects:
            predicted_obs = trajectory_planner.predict_obstacle_trajectories(world_objects, time_horizon, dt)
            candidates = trajectory_planner.generate_candidate_trajectories(v_current, time_horizon, dt)
            
            best_traj = None
            min_cost = float('inf')
            for tr in candidates:
                cost = trajectory_planner.evaluate_trajectory_cost(tr, predicted_obs, world_objects)
                if cost < min_cost:
                    min_cost = cost
                    best_traj = tr
            
            y_target = best_traj['offset'] if best_traj else 0.0
            
            # Convert lateral swerve into exact steering angle
            peak_steer = (y_target * 2.0 * math.pi * wheelbase) / (max(v_current, 0.1) * (time_horizon ** 2))
            steer_angle = peak_steer * math.sin(2 * math.pi * (dt / time_horizon))
            
            # Dynamic Velocity Control (NEVER stop completely)
            # If cost > 1,000,000, it's a critical threat. Apply brakes.
            is_critical = min_cost > 1000000
        else:
            steer_angle = 0.0
            is_critical = False
            
        # --- LONGITUDINAL CONTROLLER ---
        if is_critical:
            # We need to brake
            accel = -3.0 # m/s^2 deceleration
        else:
            # Safe, accelerate back to target speed smoothly
            if v_current < v_target:
                accel = 1.0 # m/s^2 acceleration
            else:
                accel = 0.0
                
        # Physics Step: Update speed
        v_next = v_current + accel * dt
        
        # Enforce constraints: NEVER STOP (min 2.0 m/s), never exceed target (10.0 m/s)
        if v_next <= v_min:
            v_next = v_min
            # Recalculate physical acceleration to match clamped speed
            accel = (v_next - v_current) / dt 
        elif v_next > v_target:
            v_next = v_target
            accel = (v_next - v_current) / dt
            
        v_current = v_next
        
        # --- REAL-TIME MATLAB EXPORT ---
        # Generate the instant .mat file instruction
        mat_payload = {
            'acceleration': float(accel),
            'steering': float(steer_angle),
            'v_current': float(v_current)
        }
        
        mat_filename = os.path.join(output_dir, f'frame_{frame_idx:04d}.mat')
        sio.savemat(mat_filename, mat_payload)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000.0
        total_latency += latency
        
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx:04d} | Speed: {v_current:.2f}m/s | Accel: {accel:+.2f} | Steer: {steer_angle:+.4f} | Latency: {latency:.1f}ms")
            
        frame_idx += 1
        
    cap.release()
    print("========================================")
    print(f"✅ Simulation Complete! Processed {frame_idx} frames.")
    print(f"📊 Average Frame Latency: {total_latency/max(1, frame_idx):.1f} ms")
    print(f"📁 Real-time control files stored in: {output_dir}")

if __name__ == '__main__':
    main()
