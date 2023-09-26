# Trying gaussian filter

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate
from skimage.filters import median, gaussian

# Open image
img = io.imread('data/SaltPepper.png')


# GAUSSIAN FILTER
def gaussian_filter(img, sigma):
    gauss_img = gaussian(img, sigma)
    return gauss_img


# MEDIAN FILTER
def median_filter(img, size):
    footprint = np.ones([size, size])
    med_img = median(img, footprint)
    return med_img


# Apply filters
gauss_img = gaussian_filter(img, 5)
median_img = median_filter(img, 30)

# Compare the two images
plt.subplot(1, 2, 1)
plt.imshow(gauss_img, cmap='gray')
plt.title('Gaussian filter')
plt.subplot(1, 2, 2)
plt.imshow(median_img, cmap='gray')
plt.title('Median filter')
# Add a general title
plt.suptitle('Gaussian vs Median filter')
plt.show()
