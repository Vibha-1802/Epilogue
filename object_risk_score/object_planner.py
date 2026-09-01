import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import the class rules and WorldObject from trajectory_planner
from trajectory_planner import CLASS_RULES, WorldObject, predict_obstacle_trajectories

def calculate_ttc(predicted_obs, ego_speed, time_horizon, dt=0.5):
    """
    Calculates the exact Time-To-Collision (TTC) for each object.
    Assumes Ego Vehicle travels strictly forward at ego_speed (offset = 0.0m).
    """
    ttc_results = {}
    
    for obj_id, traj in predicted_obs.items():
        ttc = float('inf')
        for state in traj:
            t = state['time']
            # Ego vehicle position at time t
            ego_x = ego_speed * t
            ego_y = 0.0 # Strict longitudinal path
            
            # Distance between Ego and Object at time t
            dist = np.sqrt((ego_x - state['x'])**2 + (ego_y - state['y'])**2)
            
            # If distance is less than the uncertainty radius + vehicle width threshold
            if dist < state['radius'] + 1.0: # 1.0m is rough ego half-width
                ttc = t
                break
                
        ttc_results[obj_id] = ttc
        
    return ttc_results

def plot_object_trajectories(objects, predicted_obs, ttc_results, ego_speed, time_horizon, dt=0.5):
    """
    Custom visualization tailored for threat assessment and TTC.
    """
    plt.figure(figsize=(12, 8))
    
    # 1. Plot Ego Vehicle Path
    ego_xs = [ego_speed * t for t in np.arange(0, time_horizon + dt, dt)]
    ego_ys = [0.0 for _ in ego_xs]
    plt.plot(ego_xs, ego_ys, 'b--', linewidth=2, label='Ego Projected Path')
    plt.plot(0, 0, 'b^', markersize=15, label='Ego Vehicle')
    
    # 2. Plot Objects and their Trajectories
    for obj in objects:
        traj = predicted_obs[obj.obj_id]
        ttc = ttc_results[obj.obj_id]
        
        # Color coding: Red if TTC < 2.0s, else Orange (Dynamic) or Gray (Static)
        if ttc < 2.0:
            color = 'red'
            status = 'CRITICAL'
        elif obj.hard_boundary or obj.name in ['pothole']:
            color = 'gray'
            status = 'STATIC'
        else:
            color = 'orange'
            status = 'DYNAMIC'
            
        # Plot center at t=0
        plt.plot(obj.x, obj.y, marker='s', color=color, markersize=10, label=f'{obj.name} (ID:{obj.obj_id})' if obj.obj_id == 1 else "")
        
        # Plot predicted expansion (uncertainty) at t=max
        final_state = traj[-1]
        circle = patches.Circle((final_state['x'], final_state['y']), final_state['radius'], 
                              linewidth=1, edgecolor=color, facecolor=color, alpha=0.1)
        plt.gca().add_patch(circle)
        
        # Draw dotted line for predicted movement path
        oxs = [p['x'] for p in traj]
        oys = [p['y'] for p in traj]
        plt.plot(oxs, oys, color=color, linestyle='-', linewidth=2, alpha=0.6)
        
        # Annotate TTC directly above object
        ttc_str = f"TTC: {ttc:.1f}s" if ttc != float('inf') else "TTC: SAFE"
        plt.text(obj.x, obj.y + 1.5, f"[{status}]\n{ttc_str}", 
                 color=color, fontsize=9, fontweight='bold', ha='center')
        
    plt.title('Object Planner: Threat Assessment & TTC')
    plt.xlabel('Longitudinal Distance (X)')
    plt.ylabel('Lateral Distance (Y)')
    plt.grid(True)
    
    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.axis('equal')
    plt.savefig('object_planner_output.png')
    print("Saved plot to object_planner_output.png")

def main():
    print("Initializing Object Planner & Threat Assessment Node...")
    
    # Mock World State (Crowded Scene from Sensor Fusion at Frame 0)
    objects = [
        WorldObject(obj_id=1, class_id=0, x=15.0, y=0.0, vx=-2.0, vy=0.0, risk_score=50), # Center Car
        WorldObject(obj_id=2, class_id=6, x=10.0, y=2.5, vx=0.0, vy=0.0, risk_score=50),  # Pedestrian
        WorldObject(obj_id=3, class_id=5, x=12.0, y=-2.5, vx=0.0, vy=0.0, risk_score=50), # Bicycle
        WorldObject(obj_id=4, class_id=0, x=20.0, y=1.0, vx=0.0, vy=0.0, risk_score=50),  # Pothole
        WorldObject(obj_id=5, class_id=0, x=25.0, y=0.0, vx=-5.0, vy=0.0, risk_score=50), # Truck
    ]
    
    time_horizon = 3.0
    ego_speed = 10.0 # 10 m/s relative speed
    
    # Simulate 5 continuous frames (e.g., polling sensors at dt = 0.5s)
    num_frames = 5
    dt = 0.5
    
    for frame in range(1, num_frames + 1):
        print(f"\n{'='*40}")
        print(f"--- PROCESSING FRAME {frame} (Simulation Time: {(frame-1)*dt:.1f}s) ---")
        
        # 1. Predict Trajectories for the current frame
        predicted_obs = predict_obstacle_trajectories(objects, time_horizon)
        
        # 2. Calculate TTC based on current positions
        ttc_results = calculate_ttc(predicted_obs, ego_speed, time_horizon)
        
        for obj in objects:
            ttc = ttc_results[obj.obj_id]
            if ttc < 2.0:
                print(f"  [!] CRITICAL THREAT: {obj.name.upper()} (ID: {obj.obj_id}) - TTC: {ttc:.1f}s")
            elif ttc != float('inf'):
                print(f"  [+] WARNING: {obj.name.upper()} (ID: {obj.obj_id}) - TTC: {ttc:.1f}s")
            else:
                print(f"  [-] SAFE: {obj.name.upper()} (ID: {obj.obj_id}) - No Collision Predicted.")
                
        # 3. Step Physics forward for the next frame
        # In a real system, sensor fusion would provide completely new object coordinates here.
        # We simulate this by mathematically driving the objects forward by their velocity.
        for obj in objects:
            if not obj.hard_boundary and obj.name not in ['pothole']:
                # The Ego vehicle is moving at 10 m/s.
                # To maintain relative coordinates (Ego is always at 0,0), we subtract Ego's movement from the objects.
                obj.x += (obj.vx - ego_speed) * dt
                obj.y += obj.vy * dt
                
    # Visualize the FINAL frame state
    plot_object_trajectories(objects, predicted_obs, ttc_results, ego_speed, time_horizon)
    print("\nSimulated Continuous Tracking Complete.")

if __name__ == "__main__":
    main()
