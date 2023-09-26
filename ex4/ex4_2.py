from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib

# Functions

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

# DATA MATRIX
all_images = glob.glob("processed_cats_100/*.jpg") # List of all images
n_images = len(all_images) # Number of images
# Read first image to get size
im = io.imread(all_images[0])
height, width, channels = im.shape
n_samples = n_images # Number of samples
n_features = height * width * channels # Number of features
data_matrix = np.zeros((n_samples, n_features)) # Data matrix
for idx in range(n_images):
    im = io.imread(all_images[idx])
    flat_img = im.flatten()
    data_matrix[idx, :] = flat_img

# PCA
print("Performing PCA")
pca = PCA(n_components=50)
pca.fit(data_matrix)

# PCA components
print("Projecting images into PCA space")
components = pca.transform(data_matrix)


# EXTREME CATS
print("Finding images with most extreme first and second components")

# Find the image with the most extreme first component
max_idx_1 = np.argmax(components[:, 0])
min_idx_1 = np.argmin(components[:, 0])
# Read the images with the most extreme first component from the data matrix
max_img_1 = data_matrix[max_idx_1, :]
min_img_1 = data_matrix[min_idx_1, :]
# Create images from the vectors
max_img_1 = create_u_byte_image_from_vector(max_img_1, height, width, channels)
min_img_1 = create_u_byte_image_from_vector(min_img_1, height, width, channels)
# Show the images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(max_img_1)
ax[1].imshow(min_img_1)
plt.show()

# Find the image with the most extreme second component
max_idx_2 = np.argmax(components[:, 1])
min_idx_2 = np.argmin(components[:, 1])
# Read the images with the most extreme second component from the data matrix
max_img_2 = data_matrix[max_idx_2, :]
min_img_2 = data_matrix[min_idx_2, :]
# Create images from the vectors
max_img_2 = create_u_byte_image_from_vector(max_img_2, height, width, channels)
min_img_2 = create_u_byte_image_from_vector(min_img_2, height, width, channels)
# Show the images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(max_img_2)
ax[1].imshow(min_img_2)
plt.show()

# Plot the first two components
print("Plotting the first two components")
fig, ax = plt.subplots()
ax.scatter(components[:, 0], components[:, 1])
ax.set_xlabel("First component")
ax.set_ylabel("Second component")
# Mark in red the most extreme cats
ax.scatter(components[max_idx_1, 0], components[max_idx_1, 1], c='r')
ax.scatter(components[min_idx_1, 0], components[min_idx_1, 1], c='r')
ax.scatter(components[max_idx_2, 0], components[max_idx_2, 1], c='r')
ax.scatter(components[min_idx_2, 0], components[min_idx_2, 1], c='r')
plt.show()




