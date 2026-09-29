import open3d as o3d
import numpy as np

WIDTH = 640
HEIGHT = 480
try:
    render = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    render.scene.set_background([0, 0, 0, 1])
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    
    render.scene.add_geometry("pcd", pcd, mat)
    img = render.render_to_image()
    print("OffscreenRenderer SUCCESS. Shape:", np.asarray(img).shape)
except Exception as e:
    print("OffscreenRenderer FAILED:", e)
