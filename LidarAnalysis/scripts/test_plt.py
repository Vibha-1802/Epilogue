import numpy as np
import matplotlib.pyplot as plt
import time

points = np.random.rand(50000, 3) * 100
colors = np.random.rand(50000, 3)

start = time.time()
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5, depthshade=False)
ax.view_init(elev=10, azim=180)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')

fig.canvas.draw()
img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.close(fig)
print("Time taken:", time.time() - start)
