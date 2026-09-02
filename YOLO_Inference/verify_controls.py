import os
import scipy.io as sio

output_dir = '/Users/pavanreddy/Epilogue/YOLO_Inference/realtime_controls'
mat_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.mat')])

min_v = float('inf')
max_v = float('-inf')
max_steer = 0.0

for f in mat_files:
    data = sio.loadmat(os.path.join(output_dir, f))
    v = data['v_current'][0][0]
    steer = data['steering'][0][0]
    
    if v < min_v: min_v = v
    if v > max_v: max_v = v
    if abs(steer) > max_steer: max_steer = abs(steer)
    
print(f"Verified {len(mat_files)} frames.")
print(f"Min Speed: {min_v:.2f} m/s")
print(f"Max Speed: {max_v:.2f} m/s")
print(f"Max Steering angle (abs): {max_steer:.4f} rad")

if min_v >= 2.0:
    print("✅ Constraints met: Vehicle never stops completely.")
else:
    print("❌ Constraints violated: Vehicle speed dropped below 2.0 m/s.")
