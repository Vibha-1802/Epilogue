import scipy.io
import numpy as np

mat_data = scipy.io.loadmat("sensor_data.mat")

print("Variables:")
for key in mat_data:
    if not key.startswith("__"):
        print(key)

radar = mat_data["radar_matrix"]

print("\n========== RADAR MATRIX ==========")
print("Python type:", type(radar))
print("Shape:", radar.shape)
print("Dtype:", radar.dtype)
print("Dtype names:", radar.dtype.names)

print("\nRaw contents:")
print(radar)

if radar.dtype.names:
    print("\n========== FIELDS ==========")
    for field in radar.dtype.names:
        value = radar[field]
        print(f"\nField: {field}")
        print("Type:", type(value))
        print("Shape:", value.shape)
        print("Value:", value)
