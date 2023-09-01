import pydicom as dicom
from skimage import io

in_dir = "data/"
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

im = ds.pixel_array
# Size of the image
print(im.shape)

# Pixel type
print(im.dtype)

# Show image
# change contrast
vmin_value = -995
vmax_value = 500
io.imshow(im, vmin=vmin_value, vmax=vmax_value, cmap='gray')
io.show()