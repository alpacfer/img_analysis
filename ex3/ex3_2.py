from skimage import color, io, measure, img_as_ubyte
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import skimage
import time
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Read image
im_org = io.imread("dark_background.png")

# Convert to gray scale
im_gray = color.rgb2gray(im_org)
im_gray = img_as_ubyte(im_gray)


# Function that applies thresholding
def threshold_image(img, threshold):
    # Do the thresholding
    img_out = img > threshold
    # Convert to ubyte
    return img_as_ubyte(img_out)


# Find the threshold using Otsu's method
thresh = threshold_otsu(im_gray)
print("Threshold: " + str(thresh))

# Apply thresholding
im_thresh = threshold_image(im_gray, thresh - 53)

# Show the result
io.imshow(im_thresh)
io.show()
