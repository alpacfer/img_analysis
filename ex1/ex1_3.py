from skimage import color, io, measure, img_as_ubyte # type: ignore
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing image
in_dir = "data/"
im_name = "dark.jpg"
im_dark = io.imread(in_dir + im_name)
im_bright = io.imread(in_dir + "bright.jpg")

# Rescaled image
im_rescaled_dark = rescale(im_dark, 0.25, anti_aliasing=True, channel_axis=2)
im_rescaled_bright = rescale(im_bright, 0.25, anti_aliasing=True, channel_axis=2)

# Gray scale
im_gray_dark = color.rgb2gray(im_rescaled_dark)
im_byte_dark = img_as_ubyte(im_gray_dark)
im_gray_bright = color.rgb2gray(im_rescaled_bright)
im_byte_bright = img_as_ubyte(im_gray_bright)

# Show the 2 histograms in the same plot
plt.hist(im_byte_dark.ravel(), bins=256, alpha=0.5, label='Dark')
plt.hist(im_byte_bright.ravel(), bins=256, alpha=0.5, label='Bright')
plt.title('Image histogram')
plt.legend(loc='upper right')
io.show()

# Explain the differences between the histograms
# The histogram of the dark image is shifted to the left, because the pixels are darker.
# The histogram of the bright image is shifted to the right, because the pixels are brighter.


