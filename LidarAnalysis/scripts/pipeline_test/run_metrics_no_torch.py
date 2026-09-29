import os
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import open3d as o3d

PCD_FILE = "../../data/raw/lidar_with_intensity_and_clusters/frame_00008.pcd"

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

print(f"Points: Traditional={N_raw}, Ours={N_fov}, Reduction={pts_red:.2f}%")
print(f"RAM: Traditional={ram_mb_t:.4f}MB, Ours={ram_mb_o:.4f}MB, Reduction={ram_red:.2f}%")
print(f"Disk: Traditional={disk_mb_t:.4f}MB, Ours={disk_mb_o:.4f}MB, Reduction={disk_red:.2f}%")
print(f"CPU_Time: Traditional={traditional_cpu_time:.4f}s, Ours={our_cpu_time:.4f}s, Reduction={cpu_red:.2f}%")
