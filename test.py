import numpy as np

# Define point
point = np.array([45, 23])

# Transformation matrix
T = np.array([[0.5, 2],
              [2, 0.8]])

# Apply transformation
new_point = np.dot(T, point)
print(new_point)

# Translation
trans = np.array([-15, 20])

# Apply translation
new_point = new_point + trans
print(new_point)
