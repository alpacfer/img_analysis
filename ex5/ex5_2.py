# BORDER HANDELING

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate


# Function to show image of a matrix in gray scale and add title
def print_img(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


# Create simple image
input_img = np.arange(25).reshape(5, 5)
print_img(input_img, 'Original image')
print(input_img, )

# Simple filter
weights = [[0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]]

# Try different border modes
# Zero padding
res_img = correlate(input_img, weights, mode='constant', cval=0)
print_img(res_img, 'Zero padding')
# Reflect padding
res_img = correlate(input_img, weights, mode='reflect')
print_img(res_img, 'Reflect padding')
# Constant padding
res_img = correlate(input_img, weights, mode='constant', cval=10)
print_img(res_img, 'Constant padding')
