# Trying mean and median filters

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate
from skimage.filters import median

# Open image
img = io.imread('data/Gaussian.png')

# MEAN FILTER
size = 30
weights = np.ones([size, size])
weights = weights / np.sum(weights)

# Apply filter
mean_img = correlate(img, weights)

# Show images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(mean_img, cmap='gray')
plt.title('Mean filter')
# Add a general title
plt.suptitle('Mean filter with size: ' + str(size))
plt.show()

# MEDIAN FILTER
size = 40
footprint = np.ones([size, size])
med_img = median(img, footprint)

# Show images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(med_img, cmap='gray')
plt.title('Median filter')
# Add a general title
plt.suptitle('Median filter with size: ' + str(size))
plt.show()
