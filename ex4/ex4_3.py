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
print("Projecting images into PCA space")
components = pca.transform(data_matrix)

# # Plot the first two components
# plt.figure(figsize=(10, 10))
# plt.plot(components[:, 0], components[:, 1], '.')
# plt.xlabel("First component")
# plt.ylabel("Second component")
# plt.show()


# SYSTESIZE NEW CATS
print("Synthesizing new cats")

# Compute the average cat
avg_cat = np.mean(data_matrix, axis=0)
avg_cat_img = create_u_byte_image_from_vector(avg_cat, height, width, channels)

# Function to create a new cat
def create_new_cat(w1, w2):
    # Create a new cat
    synth_cat = avg_cat + w1 * pca.components_[0, :] + w2 * pca.components_[1, :]
    # Create an image from the vector
    synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    return synth_cat_img


# # Try different weights and show the cats
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.imshow(create_new_cat(0, 0))
# plt.subplot(2, 2, 2)
# plt.imshow(create_new_cat(-40000, 40000))
# plt.subplot(2, 2, 3)
# plt.imshow(create_new_cat(-40000, -30000))
# plt.subplot(2, 2, 4)
# plt.imshow(create_new_cat(40000, 40000))
# plt.show()

# Modes of variation
print("Computing modes of variation")
# First principal component
m = 0
synth_cat_plus = avg_cat + 3 * np.sqrt(pca.explained_variance_[m]) * pca.components_[m, :]
synth_cat_minus = avg_cat - 3 * np.sqrt(pca.explained_variance_[m]) * pca.components_[m, :]
# Create images from the vectors
synth_cat_plus_img = create_u_byte_image_from_vector(synth_cat_plus, height, width, channels)
synth_cat_minus_img = create_u_byte_image_from_vector(synth_cat_minus, height, width, channels)
# Show the images. Minus is on the left, plus is on the right, average is in the middle
fig, ax = plt.subplots(1, 3)
ax[0].imshow(synth_cat_minus_img)
ax[1].imshow(avg_cat_img)
ax[2].imshow(synth_cat_plus_img)
plt.show()

# Second principal component
m = 1
synth_cat_plus = avg_cat + 3 * np.sqrt(pca.explained_variance_[m]) * pca.components_[m, :]
synth_cat_minus = avg_cat - 3 * np.sqrt(pca.explained_variance_[m]) * pca.components_[m, :]
# Create images from the vectors
synth_cat_plus_img = create_u_byte_image_from_vector(synth_cat_plus, height, width, channels)
synth_cat_minus_img = create_u_byte_image_from_vector(synth_cat_minus, height, width, channels)
# Show the images. Minus is on the left, plus is on the right, average is in the middle
fig, ax = plt.subplots(1, 3)
ax[0].imshow(synth_cat_minus_img)
ax[1].imshow(avg_cat_img)
ax[2].imshow(synth_cat_plus_img)
plt.show()



