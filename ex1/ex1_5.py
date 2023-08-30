from skimage import color, io, measure, img_as_ubyte # type: ignore
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing image
in_dir = "data/"
im_name = "DTUSign1.jpg"
im_org = io.imread(in_dir + im_name)

# Visualize the red component
im_red = im_org[:,:,0]
io.imshow(im_red)
plt.title('Red component')
io.show()

# Visualize the green component
im_green = im_org[:,:,1]
io.imshow(im_green)
plt.title('Green component')
io.show()

# Visualize the blue component
im_blue = im_org[:,:,2]
io.imshow(im_blue)
plt.title('Blue component')
io.show()

# The DTU Compute sign is red, so it looks bright in the R channel image and dark in the G and B channels.
# The walls of the building are white, so they look bright in all channels.

