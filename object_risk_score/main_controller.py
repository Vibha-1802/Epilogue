import time
import trajectory_planner as tp
from sensor_fusion_engine import get_fused_objects
from object_planner import predict_obstacle_trajectories, calculate_ttc

def main():
    print("==================================================")
    print("    ADAS MASTER CONTROLLER INITIALIZING...")
    print("==================================================")
    
    # 1. Initialize Sensor Fusion to get starting frame objects
    print("\n[PERCEPTION] Polling sensors & Calculating Risk...")
    world_objects = get_fused_objects()
    print(f"  -> Extracted {len(world_objects)} fused WorldObjects.")
    for obj in world_objects:
        print(f"     - {obj.name.upper()} (ID: {obj.obj_id}) | Risk Score: {obj.risk_score:.1f}/100")
    
    time_horizon = 3.0
    ego_speed = 10.0 # 10 m/s
    num_frames = 5
    dt = 0.5
    
    for frame in range(1, num_frames + 1):
        print(f"\n{'-'*50}")
        print(f"--- PROCESSING FRAME {frame} (Simulation Time: {(frame-1)*dt:.1f}s) ---")
        
        # 2. THREAT ASSESSMENT (Object Planner)
        predicted_obs = predict_obstacle_trajectories(world_objects, time_horizon)
        ttc_results = calculate_ttc(predicted_obs, ego_speed, time_horizon)
        
        aeb_trigger = False
        print("[THREAT ASSESSMENT] Analyzing TTC...")
        for obj in world_objects:
            ttc = ttc_results[obj.obj_id]
            if ttc < 2.0:
                print(f"  [!] CRITICAL THREAT: {obj.name.upper()} (ID: {obj.obj_id}) - TTC: {ttc:.1f}s")
                aeb_trigger = True
            elif ttc != float('inf'):
                print(f"  [+] WARNING: {obj.name.upper()} (ID: {obj.obj_id}) - TTC: {ttc:.1f}s")
                
        if aeb_trigger:
            print("  >>> AEB TRIGGERED BY OBJECT PLANNER <<<")
            
        # 3. ACTION PLAN (Trajectory Planner)
        print("\n[TRAJECTORY PLANNER] Evaluating swerve paths...")
        candidates = tp.generate_candidate_trajectories(ego_speed, time_horizon)
        
        best_traj = None
        min_cost = float('inf')
        
        for tr in candidates:
            cost = tp.evaluate_trajectory_cost(tr, predicted_obs, world_objects)
            if cost < min_cost:
                min_cost = cost
                best_traj = tr
                
        print(f"  >> Selected Best Safe Path: Lateral Offset {best_traj['offset']}m")
        
        # 4. ACTUATION (Simulink Export)
        print("\n[ACTUATION] Exporting to Simulink...")
        # If AEB was triggered by TTC, force the braking override in the Simulink export!
        tp.export_simulink_controls(best_traj, aeb_triggered=aeb_trigger, time_horizon=time_horizon, dt=dt)
        
        # 5. STEP PHYSICS (Simulate next frame)
        for obj in world_objects:
            if not obj.hard_boundary and obj.name not in ['pothole']:
                obj.x += (obj.vx - ego_speed) * dt
                obj.y += obj.vy * dt
                
        time.sleep(0.5) # Simulate real-time processing delay
        
    print(f"\n{'-'*50}")
    print("Master Controller Simulation Complete.")
    
if __name__ == "__main__":
    main()
