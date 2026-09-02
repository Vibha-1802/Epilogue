import os
import cv2
import time
import scipy.io as sio
import logging

# Import the modular nodes
from perception_node import YOLOPerceptor
from sensor_fusion_node import SensorFusionNode
from controller_node import VehicleControllerNode
import risk_score_calculator
import trajectory_planner

def main():
    print("Initializing Modular Vehicle Behavior Pipeline...")
    
    # 1. Configuration
    model_path = '/Users/pavanreddy/Epilogue/vehicle_behaviour_pipeline/best_parinita.pt'
    input_video = '/Users/pavanreddy/Epilogue/YOLO_Inference/matlab_data_4/cam_video.mp4'
    output_dir = '/Users/pavanreddy/Epilogue/vehicle_behaviour_pipeline/realtime_controls'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean previous output
    for f in os.listdir(output_dir):
        if f.endswith('.mat'):
            os.remove(os.path.join(output_dir, f))
            
    # 2. Instantiate Pipeline Nodes
    perceptor = YOLOPerceptor(model_path)
    fusion_node = SensorFusionNode()
    controller = VehicleControllerNode(v_initial=10.0, v_target=10.0, v_min=2.0)
    
    # 3. Video Setup
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    dt = 1.0 / fps
    time_horizon = 3.0
    
    frame_idx = 0
    total_latency = 0.0
    
    print("Starting pipeline execution loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.time()
        
        # NODE 1: Perception
        boxes = perceptor.process_frame(frame)
        
        # NODE 2: Sensor Fusion
        raw_objects = fusion_node.process_boxes(boxes, frame.shape[1])
        
        # NODE 3: Risk Assessment
        world_objects = []
        for obj_data in raw_objects:
            risk_score = risk_score_calculator.calculate_risk(
                obj_data['x'], obj_data['y'], obj_data['vx'], obj_data['vy']
            )
            obj = trajectory_planner.WorldObject(
                obj_id=obj_data['obj_id'], 
                class_id=obj_data['class_id'], 
                x=obj_data['x'], y=obj_data['y'], 
                vx=obj_data['vx'], vy=obj_data['vy'], 
                risk_score=risk_score
            )
            world_objects.append(obj)
            
        # NODE 4: Path Planning
        if world_objects:
            predicted_obs = trajectory_planner.predict_obstacle_trajectories(world_objects, time_horizon, dt)
            candidates = trajectory_planner.generate_candidate_trajectories(controller.v_current, time_horizon, dt)
            
            best_traj = None
            min_cost = float('inf')
            for tr in candidates:
                cost = trajectory_planner.evaluate_trajectory_cost(tr, predicted_obs, world_objects)
                if cost < min_cost:
                    min_cost = cost
                    best_traj = tr
            
            y_target = best_traj['offset'] if best_traj else 0.0
            is_critical = min_cost > 1000000
        else:
            y_target = 0.0
            is_critical = False
            
        # NODE 5: Vehicle Controller
        accel, steer_angle = controller.calculate_controls(y_target, is_critical, dt, time_horizon)
        
        # EXPORT: Save instructions
        mat_payload = {
            'acceleration': float(accel),
            'steering': float(steer_angle),
            'v_current': float(controller.v_current)
        }
        
        mat_filename = os.path.join(output_dir, f'frame_{frame_idx:04d}.mat')
        sio.savemat(mat_filename, mat_payload)
        
        latency = (time.time() - start_time) * 1000.0
        total_latency += latency
        
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx:04d} | Speed: {controller.v_current:.2f}m/s | Accel: {accel:+.2f} | Steer: {steer_angle:+.4f} | Latency: {latency:.1f}ms")
            
        frame_idx += 1
        
    cap.release()
    print("========================================")
    print(f"✅ Modular Pipeline Complete! Processed {frame_idx} frames.")
    print(f"📊 Average Frame Latency: {total_latency/max(1, frame_idx):.1f} ms")

if __name__ == '__main__':
    main()
