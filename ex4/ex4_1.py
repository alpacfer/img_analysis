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


def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = int(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 3
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[1 + i * 2]
        lm[i, 1] = lm_s[2 + i * 2]
    return lm


def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
    """
    Landmark based alignment of one cat image to a destination
    :param img_src: Image of source cat
    :param lm_src: Landmarks for source cat
    :param lm_dst: Landmarks for destination cat
    :return: Warped and cropped source image. None if something did not work
    """
    tform = SimilarityTransform()
    tform.estimate(lm_src, lm_dst)
    warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

    # Center of crop region
    cy = 185
    cx = 210
    # half the size of the crop box
    sz = 180
    warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
    shape = warp_crop.shape
    if shape[0] == sz * 2 and shape[1] == sz * 2:
        return img_as_ubyte(warp_crop)
    else:
        print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
        return None
    

def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    all_images = glob.glob(in_dir + "*.jpg")
    for img_idx in all_images:
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        src_img = io.imread(f"{name_no_ext}.jpg")

        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if proc_img is not None:
            io.imsave(out_name, proc_img)



def preprocess_one_cat():
    src = "data/MissingCat"
    dst = "data/ModelCat"
    out = "data/MissingCatProcessed.jpg"

    src_lm = read_landmark_file(f"{src}.jpg.cat")
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")

    src_img = io.imread(f"{src}.jpg")
    dst_img = io.imread(f"{dst}.jpg")

    src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
    if src_proc is None:
        return

    io.imsave(out, src_proc)

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src_lm[:, 0], src_lm[:, 1], '.r', markersize=12)
    ax[1].imshow(dst_img)
    ax[1].plot(dst_lm[:, 0], dst_lm[:, 1], '.r', markersize=12)
    ax[2].imshow(src_proc)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out
    
    
# # Preprocess all images
# preprocess_all_cats("training_cats_100/", "processed_cats_100")

# Create a data matrix with all images
all_images = glob.glob("processed_cats_100/*.jpg") # List of all images
n_images = len(all_images) # Number of images
# Read first image to get size
im = io.imread(all_images[0])
height, width, channels = im.shape

n_samples = n_images # Number of samples
n_features = height * width * channels # Number of features

data_matrix = np.zeros((n_samples, n_features)) # Data matrix

# Read the images and store them in the data matrix
for idx in range(n_images):
    im = io.imread(all_images[idx])
    flat_img = im.flatten()
    data_matrix[idx, :] = flat_img

# # Compute the average cat
# avg_cat = np.mean(data_matrix, axis=0)
# avg_cat_img = create_u_byte_image_from_vector(avg_cat, height, width, channels)
# # Show the average cat
# io.imshow(avg_cat_img)
# plt.show()

# Find the missing cat
# Preprocess the missing cat
# preprocess_one_cat()

# Read the missing cat
missing_cat = io.imread("data/MissingCatProcessed.jpg")
missing_cat_vec = missing_cat.flatten()

# # Subtract the missing from all cats
# sub_data_matrix = np.subtract(data_matrix, missing_cat_vec)
# # For each row, compute the sum of squared differences
# sub_distances = np.linalg.norm(sub_data_matrix, axis=1)
# # Find the index of the smallest distance
# min_idx = np.argmin(sub_distances)
# # Read the image with the smallest distance
# min_img = io.imread(all_images[min_idx]) # Most similar cat
# max_img = io.imread(all_images[np.argmax(sub_distances)]) # Most dissimilar cat

# print(f"Closest cat is {all_images[min_idx]}")

# # Show the missing cat, the most similar cat and the most dissimilar cat
# fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
# ax[0].imshow(missing_cat)
# ax[1].imshow(min_img)
# ax[2].imshow(max_img)
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Principal component analysis of the cats
print("Performing PCA")
pca = PCA(n_components=50)
pca.fit(data_matrix)

# Show the amount of variance explained by each component
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Project the cat images into pca space
print("Projecting images into PCA space")
components = pca.transform(data_matrix)

# Plot the first two components
plt.scatter(components[:, 0], components[:, 1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()