# Trying median and mean filters in salt an pepper

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate
from skimage.filters import median

# Open image
img = io.imread('data/SaltPepper.png')


# MEAN FILTER
def mean_filter(img, size):
    weights = np.ones([size, size])
    weights = weights / np.sum(weights)

    # Apply filter
    mean_img = correlate(img, weights, mode='constant', cval=0)
    return mean_img


# MEDIAN FILTER
def median_filter(img, size):
    footprint = np.ones([size, size])
    med_img = median(img, footprint)
    return med_img


# Try filters and show images
mean_img_5 = mean_filter(img, 5)
mean_img_10 = mean_filter(img, 10)
mean_img_20 = mean_filter(img, 20)
mean_img_40 = mean_filter(img, 30)

median_img_5 = median_filter(img, 5)
median_img_10 = median_filter(img, 10)
median_img_20 = median_filter(img, 20)
median_img_40 = median_filter(img, 30)

# Show images without the original
plt.subplot(2, 4, 1)
plt.imshow(mean_img_5, cmap='gray')
plt.title('MF 5x5')
plt.subplot(2, 4, 2)
plt.imshow(mean_img_10, cmap='gray')
plt.title('MF 10x10')
plt.subplot(2, 4, 3)
plt.imshow(mean_img_20, cmap='gray')
plt.title('MF 20x20')
plt.subplot(2, 4, 4)
plt.imshow(mean_img_40, cmap='gray')
plt.title('MF 40x40')
plt.subplot(2, 4, 5)
plt.imshow(median_img_5, cmap='gray')
plt.title('MeF 5x5')
plt.subplot(2, 4, 6)
plt.imshow(median_img_10, cmap='gray')
plt.title('MeF 10x10')
plt.subplot(2, 4, 7)
plt.imshow(median_img_20, cmap='gray')
plt.title('MeF 20x20')
plt.subplot(2, 4, 8)
plt.imshow(median_img_40, cmap='gray')
plt.title('MeF 40x40')
plt.suptitle('Mean and Median filters with different sizes')
plt.show()
