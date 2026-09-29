import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Introduction
nb.cells.append(nbf.v4.new_markdown_cell("""\
# Foveated Pipeline vs Traditional Pipeline: Performance Metrics

## Background
Autonomous navigation depends on the ability of a vehicle to perceive its surroundings with high precision. While 3D Lidar point clouds provide rich spatial data, processing millions of points in real-time creates immense computational bottlenecks and memory latency. Conversely, standard 2D occupancy grids lose critical height information necessary for detecting curbs, potholes, or overhanging obstacles. To balance precision and performance, there is a need for a **foveated mapping approach** — similar to human vision — where the immediate vicinity is rendered in high detail for safety, and distant areas are simplified to reduce the processing load.

## Description
This notebook evaluates a deep learning pipeline that transforms raw Lidar point clouds into a variable resolution 2.5D grid (an elevation map with semantic layers). It compares the **Traditional Approach** (processing the full, dense 3D point cloud) with **Our Approach** (foveated, multi-resolution 2.5D grid processing).

We measure the following metrics:
- **Disk Usage**: Storage size of the processed data representation.
- **RAM Usage**: Memory footprint of the data structures.
- **Computation Time (CPU Preprocessing)**: Time taken for rasterization and KNN search.
- **Computation Time (Inference)**: Time taken for the neural network forward pass.
"""))

# Setup
nb.cells.append(nbf.v4.new_code_cell("""\
import os
import time
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree

print("Environment setup successful.")

PCD_FILE = "../../data/raw/lidar_with_intensity_and_clusters/frame_00008.pcd"
if not os.path.exists(PCD_FILE):
    print(f"File not found: {PCD_FILE}. Please update the path.")
"""))

# Mock PointSegNet
nb.cells.append(nbf.v4.new_markdown_cell("""\
## Define Neural Network Architecture (Simulation)
We define a simulated `PointSegNet` inference engine that mimics the computational complexity (matrix multiplications) of PointNet++ / PointSegNet. It processes the full dense point cloud in the traditional approach, and the reduced, foveated point cloud in our approach.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
class PointSegNetSim:
    def __init__(self, in_channels=6, num_classes=3):
        # Initialize random weights mimicking Conv layers
        self.w1 = np.random.randn(in_channels * 2, 64)
        self.w2 = np.random.randn(64, 128)
        self.w3 = np.random.randn(128, 64)
        self.w4 = np.random.randn(64, num_classes)
        
    def gather_neighbors(self, features, neighbor_idx):
        return features[neighbor_idx]

    def forward(self, features, neighbor_idx):
        # 1. Gather & combine
        gathered = self.gather_neighbors(features, neighbor_idx)
        center = np.repeat(features[:, np.newaxis, :], neighbor_idx.shape[1], axis=1)
        combined = np.concatenate([center, gathered - center], axis=-1)  # [N, K, 2C]
        
        # 2. Shared MLPs (Conv2D equivalents)
        x = np.dot(combined, self.w1)
        x = np.maximum(0, x) # ReLU
        
        x = np.dot(x, self.w2)
        x = np.maximum(0, x) # ReLU
        
        # 3. Max Pooling
        x = np.max(x, axis=1) # [N, 128]
        
        # 4. Classification Head (Conv1D equivalents)
        x = np.dot(x, self.w3)
        x = np.maximum(0, x)
        logits = np.dot(x, self.w4)
        
        return logits

model = PointSegNetSim()
print("Model initialized.")
"""))

# Foveated Grid
nb.cells.append(nbf.v4.new_code_cell("""\
class MultiResGrid:
    def __init__(self, ring_bounds=(50.0, 85.0, 100.0), cell_sizes=(0.05, 0.25, 0.50)):
        self.ring_bounds = np.array(ring_bounds, dtype=float)
        self.cell_sizes = np.array(cell_sizes, dtype=float)
        self.n_rings = len(ring_bounds)
        
    def assign_ring(self, r):
        idx = np.searchsorted(self.ring_bounds, r, side='left')
        idx[idx >= self.n_rings] = -1
        return idx
        
    def rasterize(self, xyz, intensity):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.sqrt(x**2 + y**2)
        ring_idx = self.assign_ring(r)
        
        ring_representations = {}
        for k in range(self.n_rings):
            mask = ring_idx == k
            x_k, y_k, z_k, int_k = x[mask], y[mask], z[mask], intensity[mask]
            
            cs = self.cell_sizes[k]
            if not mask.any(): continue
                
            ix = np.floor(x_k / cs).astype(int)
            iy = np.floor(y_k / cs).astype(int)
            
            df = pd.DataFrame({'z': z_k, 'intensity': int_k, 'ix': ix, 'iy': iy})
            agg = df.groupby(['ix', 'iy']).agg(
                z_mean=('z', 'mean'),
                representative_intensity=('intensity', 'max')
            ).reset_index()
            
            agg['x_center'] = (agg['ix'] + 0.5) * cs
            agg['y_center'] = (agg['iy'] + 0.5) * cs
            ring_representations[k] = agg
            
        return ring_representations

def save_and_measure_disk(features, labels, neighbor_idx, filename):
    np.savez_compressed(filename, features=features, labels=labels, neighbor_idx=neighbor_idx)
    return os.path.getsize(filename)
"""))

# Traditional Pipeline
nb.cells.append(nbf.v4.new_markdown_cell("""\
---
## 1. Traditional Pipeline Execution
Processing 100% of the raw 3D point cloud points.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
print("--- TRADITIONAL PIPELINE ---")

t_pcd = o3d.t.io.read_point_cloud(PCD_FILE)
points = t_pcd.point.positions.numpy()
valid_mask = np.isfinite(points).all(axis=1)
raw_coords = points[valid_mask]
raw_intensity = t_pcd.point.intensity.numpy()[valid_mask].flatten() if 'intensity' in t_pcd.point else np.zeros(len(raw_coords))

N_raw = len(raw_coords)
print(f"Loaded {N_raw} raw points.")

# Metric 1: RAM
traditional_ram = raw_coords.nbytes + raw_intensity.nbytes

# Preprocessing: Features & KNN
start_cpu = time.time()
x, y, z = raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)
theta = np.arctan2(y, x)
raw_features = np.column_stack((x, y, z, raw_intensity, r, theta)).astype(np.float32)

tree_raw = cKDTree(raw_coords)
_, raw_neighbor_idx = tree_raw.query(raw_coords, k=16, workers=-1)
traditional_cpu_time = time.time() - start_cpu
print(f"KNN Computation Time: {traditional_cpu_time:.4f} s")

# Metric 2: Disk Size
traditional_disk = save_and_measure_disk(raw_features, np.zeros(N_raw, dtype=np.int32), raw_neighbor_idx, 'traditional_data.npz')

# Inference
start_infer = time.time()
_ = model.forward(raw_features, raw_neighbor_idx)
traditional_infer_time = time.time() - start_infer
print(f"Inference Time: {traditional_infer_time:.4f} s")
"""))

# Our Approach
nb.cells.append(nbf.v4.new_markdown_cell("""\
---
## 2. Our Approach Execution (Foveated Pipeline)
Processing reduced points via the MultiResGrid.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
print("--- OUR APPROACH (FOVEATED) ---")

start_cpu_our = time.time()
# 1. Rasterization
grid = MultiResGrid()
ring_reps = grid.rasterize(raw_coords, raw_intensity)

fov_coords_list, fov_int_list = [], []
for k, df in ring_reps.items():
    if len(df) > 0:
        fov_coords_list.append(df[['x_center', 'y_center', 'z_mean']].values)
        fov_int_list.append(df['representative_intensity'].values)

foveated_coords = np.vstack(fov_coords_list).astype(np.float32)
foveated_intensity = np.concatenate(fov_int_list).astype(np.float32)

N_fov = len(foveated_coords)
print(f"Reduced points after foveated rasterization: {N_fov}")

# Metric 1: RAM
our_ram = foveated_coords.nbytes + foveated_intensity.nbytes

# 2. Preprocessing: Features & KNN
x_f, y_f, z_f = foveated_coords[:, 0], foveated_coords[:, 1], foveated_coords[:, 2]
r_f = np.sqrt(x_f**2 + y_f**2 + z_f**2)
theta_f = np.arctan2(y_f, x_f)
foveated_features = np.column_stack((x_f, y_f, z_f, foveated_intensity, r_f, theta_f)).astype(np.float32)

tree_fov = cKDTree(foveated_coords)
_, foveated_neighbor_idx = tree_fov.query(foveated_coords, k=16, workers=-1)
our_cpu_time = time.time() - start_cpu_our
print(f"Rasterization + KNN Computation Time: {our_cpu_time:.4f} s")

# Metric 2: Disk Size
our_disk = save_and_measure_disk(foveated_features, np.zeros(N_fov, dtype=np.int32), foveated_neighbor_idx, 'our_data.npz')

# 3. Inference
start_infer_our = time.time()
_ = model.forward(foveated_features, foveated_neighbor_idx)
our_infer_time = time.time() - start_infer_our
print(f"Inference Time: {our_infer_time:.4f} s")
"""))

# Comparison
nb.cells.append(nbf.v4.new_markdown_cell("""\
---
## 3. Comparison Metrics & Results
The final results comparing Disk, RAM, and Computation Time between the two methods.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
def get_reduction(traditional, ours):
    if traditional == 0: return 0
    return ((traditional - ours) / traditional) * 100

print("="*85)
print("                       PERFORMANCE METRICS COMPARISON")
print("="*85)

print(f"{'Metric':<30} | {'Traditional Approach':<20} | {'Our Approach':<15} | {'Reduction %':<10}")
print("-" * 85)

pts_red = get_reduction(N_raw, N_fov)
print(f"{'Number of Points':<30} | {N_raw:<20} | {N_fov:<15} | {pts_red:.2f}%")

ram_mb_t = traditional_ram / (1024*1024)
ram_mb_o = our_ram / (1024*1024)
ram_red = get_reduction(traditional_ram, our_ram)
print(f"{'RAM Usage (MB)':<30} | {ram_mb_t:<20.4f} | {ram_mb_o:<15.4f} | {ram_red:.2f}%")

disk_mb_t = traditional_disk / (1024*1024)
disk_mb_o = our_disk / (1024*1024)
disk_red = get_reduction(traditional_disk, our_disk)
print(f"{'Disk Usage (Cache) (MB)':<30} | {disk_mb_t:<20.4f} | {disk_mb_o:<15.4f} | {disk_red:.2f}%")

cpu_red = get_reduction(traditional_cpu_time, our_cpu_time)
print(f"{'CPU Time (Raster+KNN) (s)':<30} | {traditional_cpu_time:<20.4f} | {our_cpu_time:<15.4f} | {cpu_red:.2f}%")

gpu_red = get_reduction(traditional_infer_time, our_infer_time)
print(f"{'Inference Time (s)':<30} | {traditional_infer_time:<20.4f} | {our_infer_time:<15.4f} | {gpu_red:.2f}%")

print("="*85)
print("\\nKEY TAKEAWAYS FOR PRESENTATION:")
print(f"1. Data Volume reduced by {pts_red:.1f}%, directly mitigating memory bottlenecks.")
print(f"2. Disk storage for precomputed datasets is reduced by {disk_red:.1f}%.")
print(f"3. Neural Network inference time decreased by {gpu_red:.1f}%, drastically improving FPS for real-time systems.")
"""))

with open('/Users/pavanreddy/Documents/MATLAB/LidarAnalysis/scripts/pipeline_test/metrics_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
