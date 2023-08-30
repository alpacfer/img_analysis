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

# Black rectangle in the image
im_org[500:1000, 800:1500, :] = 0
io.imshow(im_org)
io.show()

# Save the image
io.imsave(in_dir + "DTUSign1_black.png", im_org)
