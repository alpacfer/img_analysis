from skimage import color, io, measure, img_as_ubyte # type: ignore
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image
im_org = io.imread(in_dir + im_name)

# # Image size
# print(im_org.shape)

# # Pixel type
# print(im_org.dtype)

# # Show image
# io.imshow(im_org)
# plt.title('Metacarpal image')
# io.show()

# # Color maps
# io.imshow(im_org, cmap="cool")
# plt.title('Metacarpal image (with colormap)')
# io.show()

# # Gray Scale (manual)
# io.imshow(im_org, vmin=20, vmax=170)
# plt.title('Metacarpal image (gray scale)')
# io.show()


# EXCERCISE 7: Gray Scale (automatic)
# x_size, y_size = im_org.shape
# # Recorrer toda la imagen y encontrar el máximo y mínimo
# max = im_org[0,0]
# min = im_org[0,0]

# for row in range(x_size):
#     for column in range(y_size):
#         pixel = im_org[row, column]
#         if pixel > max:
#             max = pixel
#         if pixel < min:
#             min = pixel
# io.imshow(im_org, vmin = min, vmax = max)
# plt.title('Metacarpal image (gray scale)')
# io.show()


# # HISTOGRAM
# plt.hist(im_org.ravel(), bins=256)
# plt.title('Image histogram')
# io.show()

# # Most common range of intensities
# h = plt.hist(im_org.ravel(), bins=256)

# max_intensity = 0
# for bin in range(256):
#     count = h[0][bin]
#     if count > max_intensity:
#         max_intensity = count
#         bin_max = bin

# print(bin_max, max_intensity)

# # EXERCISE 10
# r = 110
# c = 90
# im_val = im_org[r, c]
# print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")


# # EXERCISE 11
# im_org[:, :30] = 0
# io.imshow(im_org)
# io.show()

# # EXERCISE 12
# mask = im_org > 150
# io.imshow(mask)
# io.show()

# EXERCISE 13
# mask = im_org > 150
# im_org[mask] = 255
# io.imshow(im_org)
# io.show()

# EXERCISE 14
in_dir = "data/"
im_name = "ardeche.jpg"

# Read the image
im_org = io.imread(in_dir + im_name)
# io.imshow(im_org)
# io.show()
# print(im_org.shape)

# EXERCISE 15
r = 110
c = 90
r_color, g_color, b_color = im_org[r,c]
print(r_color, g_color, b_color)

# EXERCISE 16
rows = im_org.shape[0]
r_2 = int(rows/2)

im_org[:r_2, :, 0] = 0  
im_org[:r_2, :, 1] = 255
im_org[:r_2, :, 2] = 0  

io.imshow(im_org)
io.show()
