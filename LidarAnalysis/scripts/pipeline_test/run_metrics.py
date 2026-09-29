import os
import time
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PCD File path
PCD_FILE = "../../data/raw/lidar_with_intensity_and_clusters/frame_00008.pcd"

class PointSegNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=3):
        super(PointSegNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, num_classes, kernel_size=1)
        
    def gather_neighbors(self, features, neighbor_idx):
        B, N, C = features.shape
        batch_indices = torch.arange(B, dtype=torch.long, device=features.device).view(B, 1, 1)
        return features[batch_indices, neighbor_idx, :]

    def forward(self, features, neighbor_idx):
        gathered = self.gather_neighbors(features, neighbor_idx)
        center = features.unsqueeze(2).expand(-1, -1, neighbor_idx.shape[-1], -1)
        combined = torch.cat([center, gathered - center], dim=-1)
        x = combined.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, dim=3)[0]
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv4(x).permute(0, 2, 1)

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

if not os.path.exists(PCD_FILE):
    print(f"File not found: {PCD_FILE}")
    exit(1)

model = PointSegNet().to(device)
model.eval()

t_pcd = o3d.t.io.read_point_cloud(PCD_FILE)
points = t_pcd.point.positions.numpy()
valid_mask = np.isfinite(points).all(axis=1)
raw_coords = points[valid_mask]
raw_intensity = t_pcd.point.intensity.numpy()[valid_mask].flatten() if 'intensity' in t_pcd.point else np.zeros(len(raw_coords))

N_raw = len(raw_coords)
traditional_ram = raw_coords.nbytes + raw_intensity.nbytes

start_cpu = time.time()
x, y, z = raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)
theta = np.arctan2(y, x)
raw_features = np.column_stack((x, y, z, raw_intensity, r, theta)).astype(np.float32)

tree_raw = cKDTree(raw_coords)
_, raw_neighbor_idx = tree_raw.query(raw_coords, k=16, workers=-1)
raw_neighbor_idx = raw_neighbor_idx.astype(np.int32)
traditional_cpu_time = time.time() - start_cpu

traditional_disk = save_and_measure_disk(raw_features, np.zeros(N_raw, dtype=np.int32), raw_neighbor_idx, 'traditional_data.npz')

features_t = torch.tensor(raw_features).unsqueeze(0).to(device)
neighbor_idx_t = torch.tensor(raw_neighbor_idx).unsqueeze(0).to(device)

start_gpu = time.time()
with torch.no_grad():
    _ = model(features_t, neighbor_idx_t)
    if device.type == 'cuda': torch.cuda.synchronize()
traditional_gpu_time = time.time() - start_gpu

# OUR APPROACH
start_cpu_our = time.time()
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

our_ram = foveated_coords.nbytes + foveated_intensity.nbytes

x_f, y_f, z_f = foveated_coords[:, 0], foveated_coords[:, 1], foveated_coords[:, 2]
r_f = np.sqrt(x_f**2 + y_f**2 + z_f**2)
theta_f = np.arctan2(y_f, x_f)
foveated_features = np.column_stack((x_f, y_f, z_f, foveated_intensity, r_f, theta_f)).astype(np.float32)

tree_fov = cKDTree(foveated_coords)
_, foveated_neighbor_idx = tree_fov.query(foveated_coords, k=16, workers=-1)
foveated_neighbor_idx = foveated_neighbor_idx.astype(np.int32)
our_cpu_time = time.time() - start_cpu_our

our_disk = save_and_measure_disk(foveated_features, np.zeros(N_fov, dtype=np.int32), foveated_neighbor_idx, 'our_data.npz')

features_t_fov = torch.tensor(foveated_features).unsqueeze(0).to(device)
neighbor_idx_t_fov = torch.tensor(foveated_neighbor_idx).unsqueeze(0).to(device)

start_gpu_our = time.time()
with torch.no_grad():
    _ = model(features_t_fov, neighbor_idx_t_fov)
    if device.type == 'cuda': torch.cuda.synchronize()
our_gpu_time = time.time() - start_gpu_our

def get_reduction(traditional, ours):
    if traditional == 0: return 0
    return ((traditional - ours) / traditional) * 100

pts_red = get_reduction(N_raw, N_fov)
ram_mb_t = traditional_ram / (1024*1024)
ram_mb_o = our_ram / (1024*1024)
ram_red = get_reduction(traditional_ram, our_ram)
disk_mb_t = traditional_disk / (1024*1024)
disk_mb_o = our_disk / (1024*1024)
disk_red = get_reduction(traditional_disk, our_disk)
cpu_red = get_reduction(traditional_cpu_time, our_cpu_time)
gpu_red = get_reduction(traditional_gpu_time, our_gpu_time)

print("METRICS_OUTPUT_START")
print(f"Points: Traditional={N_raw}, Ours={N_fov}, Reduction={pts_red:.2f}%")
print(f"RAM: Traditional={ram_mb_t:.4f}MB, Ours={ram_mb_o:.4f}MB, Reduction={ram_red:.2f}%")
print(f"Disk: Traditional={disk_mb_t:.4f}MB, Ours={disk_mb_o:.4f}MB, Reduction={disk_red:.2f}%")
print(f"CPU_Time: Traditional={traditional_cpu_time:.4f}s, Ours={our_cpu_time:.4f}s, Reduction={cpu_red:.2f}%")
print(f"GPU_Time: Traditional={traditional_gpu_time:.4f}s, Ours={our_gpu_time:.4f}s, Reduction={gpu_red:.2f}%")
print("METRICS_OUTPUT_END")
