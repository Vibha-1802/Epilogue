import open3d as o3d
import sys

file_path = "../data/raw/lidar_with_intensity_and_clusters/frame_00010.pcd"
t_pcd = o3d.t.io.read_point_cloud(file_path)

print("Available point attributes:")
print(t_pcd.point)
