# Filters with real images

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate
from skimage.filters import median, gaussian

# Open image
img = io.imread('data/car.png')
# Convert to gray scale
img = img_as_ubyte(img[:, :, 0])


# GAUSSIAN FILTER
def gaussian_filter(img, sigma):
    gauss_img = gaussian(img, sigma)
    return gauss_img


# MEDIAN FILTER
def median_filter(img, size):
    footprint = np.ones([size, size])
    med_img = median(img, footprint)
    return med_img


# MEAN FILTER
def mean_filter(img, size):
    weights = np.ones([size, size])
    weights = weights / np.sum(weights)
    mean_img = correlate(img, weights, mode='constant', cval=0)
    return mean_img


# Compare the the three filters
gauss_img = gaussian_filter(img, 5)
median_img = median_filter(img, 30)
mean_img = mean_filter(img, 30)

# Show only gaussian filter
plt.plot(1, 3, 1)
plt.imshow(gauss_img, cmap='gray')
plt.title('Gaussian filter')
plt.show()

# Show only median filter
plt.plot(1, 3, 1)
plt.imshow(median_img, cmap='gray')
plt.title('Median filter')
plt.show()

# Show only mean filter
plt.plot(1, 3, 1)
plt.imshow(mean_img, cmap='gray')
plt.title('Mean filter')
plt.show()
