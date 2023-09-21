from skimage import color, io, measure, img_as_ubyte
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import skimage
import time
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Read image
im_org = io.imread("vertebra.png")

# Histogram
plt.hist(im_org.ravel(), bins=256)  # Bimodal histogram
plt.title('Image histogram')
io.show()

# Minimum and maximum values
min = im_org.min()
max = im_org.max()
print("Min: " + str(min))
print("Max: " + str(max))

# Convert to float
im_float = img_as_float(im_org)
# Maximum and minimum values
min = im_float.min()
max = im_float.max()
print("Min: " + str(min))
print("Max: " + str(max))

# Converto to ubyte
im_ubyte = img_as_ubyte(im_float)
# Maximum and minimum values
min = im_ubyte.min()
max = im_ubyte.max()
print("Min: " + str(min))
print("Max: " + str(max))


# Histogram stretching
def histogram_stretch(img_in):
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0

    # Linear histogram stretching equation
    img_out = (max_desired - min_desired) / (max_val - min_val) * (img_float - min_val) + min_desired

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)


# Try histogram stretching with the original image
im_stretched = histogram_stretch(im_org)

# Show both images
io.imshow(im_org)
plt.title('Original image')
io.show()
io.imshow(im_stretched)
plt.title('Stretched image')
io.show()


# Gamma mapping
def gamma_map(img, gamma):
    # Convert to float
    img_float = img_as_float(img)
    # Do the gamma mapping
    img_out = img_float ** gamma
    # Convert to ubyte
    return img_as_ubyte(img_out)


# Try gamma mapping with the original image
im_gamma = gamma_map(im_org, 2)

# Show image
io.imshow(im_gamma)
plt.title('Gamma image')
io.show()


# Image segmentation by thresholding
def threshold_image(img, threshold):
    # Do the thresholding
    img_out = img > threshold
    # Convert to ubyte
    return img_as_ubyte(img_out)


# Automatic thresholding by Otsu's method
otsu_threshold = threshold_otsu(im_org)
print("Otsu threshold: " + str(otsu_threshold))

# Try thresholding with the original image
im_threshold = threshold_image(im_org, otsu_threshold)

# Show image
io.imshow(im_threshold)
plt.title('Threshold image')
io.show()
