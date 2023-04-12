import numpy as np

# Define the principal point (cx, cy) and focal length (fx, fy)
cx, cy = 946.911, 561.314
fx, fy = 1387.53, 1388.32

# Create the intrinsic camera matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Save K to a npz file
np.savez('../code/catkin_ws/src/camera_calibration/src/intrinsic_camera_calibration_data/factory_settings_1920.npz', K=K)

print("Intrinsic camera matrix K:")
print(K)