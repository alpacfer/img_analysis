from skimage import color, io, measure, img_as_ubyte  # type: ignore
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing image
in_dir = "data/"
im_name = "metacarpals.png"
im_org = io.imread(in_dir + im_name)

# # Show image
# io.imshow(im_org)
# plt.title('Metacarpal image')
# io.show()

# Mask
threshold = 130
im_mask = im_org > threshold

# Show mask and histogram together
fig, ax = plt.subplots(1, 2)
ax[0].imshow(im_mask)
ax[0].set_title('Mask')
ax[1].hist(im_org.ravel(), bins=256)
ax[1].set_title('Metacarpal image histogram')
# Show a vertical line at the threshold
ax[1].axvline(130, color='r', linestyle='dashed', linewidth=1)
# Add more space between the two plots
fig.tight_layout()
plt.show()

# Copy the original image
im_mask_color = np.copy(im_org)
# Convert the mask to RGB
im_mask_color = color.gray2rgb(im_mask_color)
# Color the mask blue
im_mask_color[im_mask] = [0, 0, 255]
# Show the mask
io.imshow(im_mask_color)
plt.title('Mask blue')
io.show()

# Save the mask
io.imsave(in_dir + "metacarpals_mask.png", im_mask_color)
