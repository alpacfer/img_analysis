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
import random


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
print("Creating data matrix")
all_images = glob.glob("processed_cats/*.jpg") # List of all images
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

# Average cat
print("Computing average cat")
avg_cat = np.mean(data_matrix, axis=0)
avg_cat_img = create_u_byte_image_from_vector(avg_cat, height, width, channels)


def synthesize_cats(n_synthesized, n_components_to_use):
    synthesized_cats = []
    for i in range(n_synthesized):
        synth_cat = avg_cat
        for idx in range(n_components_to_use):
            w = random.uniform(-1, 1) * 3 * np.sqrt(pca.explained_variance_[idx])
            synth_cat = synth_cat + w * pca.components_[idx, :]        
            
        synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)

        synthesized_cats.append(synth_cat_img)
    return synthesized_cats

cats = synthesize_cats(10, 50)
# Show synthesized cats
fig, ax = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(cats[i * 5 + j])
        ax[i, j].axis('off')
plt.show()
