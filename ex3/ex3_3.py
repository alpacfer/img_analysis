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

# Show image
io.imshow(im_org)
io.show()


# Function that detects a blue DTU sign
def detect_dtu_sign(img):
    r_comp = img[:, :, 0]
    g_comp = img[:, :, 1]
    b_comp = img[:, :, 2]

    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
                (b_comp > 180) & (b_comp < 200)

    return segm_blue


# Function in which I can select the color I want to detect

def detect_dtu_sign_pro(img, color):
    # 0 for red, 1 for green, 2 for blue
    r_comp = img[:, :, 0]
    g_comp = img[:, :, 1]
    b_comp = img[:, :, 2]

    if color == 0:  # Red
        segm = (r_comp > 150) & (g_comp < 70) & (b_comp < 70)

    elif color == 1:  # Green
        segm = (r_comp < 70) & (g_comp > 70) & (b_comp < 70)

    elif color == 2:  # Blue
        segm = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
               (b_comp > 180) & (b_comp < 200)

    return segm


# Try all three colors
im_red = detect_dtu_sign_pro(im_org, 0)
im_green = detect_dtu_sign_pro(im_org, 1)
im_blue = detect_dtu_sign_pro(im_org, 2)

# Show all three images
io.imshow(im_red)
io.show()
io.imshow(im_green)
io.show()
io.imshow(im_blue)
io.show()
