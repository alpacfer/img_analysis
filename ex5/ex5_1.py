from scipy.ndimage import correlate
import numpy as np

# Create simple image
input_img = np.arange(25).reshape(5, 5)
print(input_img)

# Simple filter
weights = [[0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]]

# Correlate image with weights
res_img = correlate(input_img, weights)
# Value of the position 3,3
print(input_img[3, 3])
print(res_img)
print(res_img[3, 3])
# Explanation: 12*0 + 13*1 + 14*0 + 17*1 + 18*2 + 19*1 + 22*0 + 23*1 + 24*0 = 108
