from skimage import color, io, measure, img_as_ubyte
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import skimage
import time
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Read image
im_org = io.imread("DTUSigns2.jpg")

# Color thresholding in the HSV color space
hsv_img = color.rgb2hsv(im_org)
hue_img = hsv_img[:, :, 0]
value_img = hsv_img[:, :, 2]
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
ax0.imshow(im_org)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(value_img)
ax2.set_title("Value channel")
ax2.axis('off')

fig.tight_layout()
io.show()
