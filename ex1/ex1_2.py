from skimage import color, io, measure, img_as_ubyte # type: ignore
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing image
in_dir = "data/"
im_name = "cool.jpg"
im_org = io.imread(in_dir + im_name)

# Rescaled image
im_rescaled = rescale(im_org, 0.25, anti_aliasing=True, channel_axis=2)

# # Type of pixels
# print(im_org.dtype)

# # Show image
# io.imshow(im_org)
# plt.title('Cool image')
# io.show()

# # Resized image
# im_resized = resize(im_org, (im_org.shape[0] // 6, im_org.shape[1] // 2), anti_aliasing=True)

# # Show image
# io.imshow(im_resized)
# plt.title('Resized image')
# io.show()

# # Specify the number of columns
# columns = 400
# # Aspect ratio of the image
# ratio = im_org.shape[1] / im_org.shape[0]

# im_resized = resize(im_org, (int(columns/ratio), columns), anti_aliasing=True)
# io.imshow(im_resized)
# io.show()

# Gray scale
im_gray = color.rgb2gray(im_org)
im_byte = img_as_ubyte(im_gray)

# Histogram
plt.hist(im_byte.ravel(), bins=256)
plt.title('Image histogram')
io.show()
