from skimage import color, io, measure, img_as_ubyte # type: ignore
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing image
in_dir = "data/"
im_name = "bright_dark.jpg"
im_bright_dark = io.imread(in_dir + im_name)

# Rescaled image
im_rescaled = rescale(im_bright_dark, 0.25, anti_aliasing=True, channel_axis=2)

# Gray scale
im_gray = color.rgb2gray(im_rescaled)
im_byte = img_as_ubyte(im_gray)

# Histogram
plt.hist(im_byte.ravel(), bins=256)
plt.title('Image histogram')
io.show()

