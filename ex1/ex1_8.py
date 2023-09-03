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

# Get a profile line
p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show()

# Show the profile line on the image
plt.figure()
plt.imshow(im_org, cmap='gray')
# Plot the profile line (source and distination points)
plt.plot(77, 342, 'ro')
plt.plot(160, 320, 'ro')
# Plot the profile line
plt.plot([77, 160], [342, 320], 'r-')
plt.show()

# Landscape view
in_dir = "data/"
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)  # type: ignore
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
