## Exercise 8 - Cats, Cats, and EigenCats
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

#%% Prepocessing
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
    dst = "ex8-CatsCatsCats/data/ModelCat"
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
    src = "ex8-CatsCatsCats/data/MissingCat"
    dst = "ex8-CatsCatsCats/data/ModelCat"
    out = "ex8-CatsCatsCats/data/MissingCatProcessed.jpg"

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

#%% Ex 1) Preprocess all image in the training set.
in_dir = os.getcwd() + "/ex8-CatsCatsCats/cats/training_cats_100/"
out_dir = os.getcwd() + "/ex8-CatsCatsCats/cats/training_cats_100_preprocessed/"
#preprocess_all_cats(in_dir, out_dir)

#%% Ex 2) Compute the data matrix.
# Find image files in the preprocessed folder
cat_faces = glob.glob("ex8-CatsCatsCats/cats/training_cats_100_preprocessed/*.jpg")
cat_face_im = []
for img_idx in cat_faces:
    name_no_ext = os.path.splitext(img_idx)[0]
    src_img = io.imread(f"{name_no_ext}.jpg")
    cat_face_im.append(src_img)

# Read the first photo and use that to find the height and width of the photos
height, width, channels = cat_face_im[0].shape
n_features = height * width * channels # Number of features
n_samples = len(cat_face_im) # Number of samples

# Empty matrix init
data_matrix = np.zeros((n_samples, n_features))

# Flatten images
cat_face_im_flat = [im.flatten() for im in cat_face_im]
for idx in range(len(cat_face_im_flat)):
    data_matrix[idx, :] = cat_face_im_flat[idx]

#%% Ex 3) Compute the average cat.
cat_mean = np.mean(data_matrix, axis=0)

#%% Ex 4) Visualize the Mean Cat
cat_mean_im = create_u_byte_image_from_vector(cat_mean, height, width, channels)
io.imshow(cat_mean_im)
plt.title('Mean cat image')
io.show()

#%% Ex 5) Decide that you quickly buy a new cat that looks very much like the 
# missing cat - so nobody notices
# SIR YES SIR!!

#%% Ex 6) Use the preprocess_one_cat function to preprocess the photo of the 
# poor missing cat
preprocess_one_cat()

#%% Ex 7) Flatten the pixel values of the missing cat so it becomes a vector 
# of values.
missing_cat = io.imread("ex8-CatsCatsCats/data/MissingCatProcessed.jpg")
missing_cat_flat = missing_cat.flatten()

io.imshow(missing_cat)
plt.title('Missing cat image')
io.show()

#%% Ex 8) Subtract you missing cat data from all the rows in the data_matrix 
# and for each row compute the sum of squared differences. This can for example 
# be done by:
sub_data = data_matrix - missing_cat_flat
sub_distances = np.linalg.norm(sub_data, axis=1)

#%% Ex 9) Find the cat that looks most like your missing cat by finding the 
# cat, where the SSD is smallest. You can for example use np.argmin.
SSD_min = np.argmin(sub_distances)

#%% Ex 10) Extract the found cat
found_cat = data_matrix[SSD_min]
found_cat_im = create_u_byte_image_from_vector(found_cat, height, width, channels)
io.imshow(found_cat_im)
plt.title('Most similar cat image')
io.show()

#%% Ex 11) Find the least lookalike cat!!
SSD_max = np.argmax(sub_distances)
found_cat_least = data_matrix[SSD_max]
found_cat_least_im = create_u_byte_image_from_vector(found_cat_least, height, width, channels)
io.imshow(found_cat_least_im)
plt.title('Most similar cat image')
io.show()


#%% Ex 12) PCA
print("Computing PCA")
cats_pca = PCA(n_components=50)
cats_pca.fit(data_matrix)

#%% Ex 13) Plot the amount of the total variation explained by each component 
# as function of the component number.
explained_variance_ratio  = cats_pca.explained_variance_ratio_
component_numbers = range(1, len(explained_variance_ratio) + 1)

plt.plot(component_numbers, explained_variance_ratio)
plt.title('Explained Variance Ratio by Component Number')
plt.xlabel('Component Number')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

#%% Ex 14) How much of the total variation is explained by the first component?
print(f"First component explains variation by {explained_variance_ratio[0]*100:.2f} %")

#%% Ex 15) Project the cat images into PCA space
components = cats_pca.transform(data_matrix)

#%% Ex 16) Plot the PCA space by plotting all the cats first and second PCA 
# coordinates in a (x, y) plot
pc_1 = components[:, 0]
pc_2 = components[:, 1]

plt.scatter(pc_2, pc_1)
plt.title('1st and 2nd PCA space of all cats')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

#%% Ex 17)
cat_pc1_min = data_matrix[np.argmin(pc_1)]
cat_pc1_max = data_matrix[np.argmax(pc_1)]
cat_pc2_min = data_matrix[np.argmin(pc_2)] 
cat_pc2_max = data_matrix[np.argmax(pc_2)]


extreme_cats_data = [cat_pc1_min, cat_pc1_max, cat_pc2_min, cat_pc2_max]
for i, cat_data in enumerate(extreme_cats_data):
    plt.subplot(1, 4, i + 1)
    cat_image = create_u_byte_image_from_vector(cat_data, height, width, channels)
    plt.imshow(cat_image)
    plt.title(f'Cat {i+1}')
    plt.axis('off')
plt.show()

pca_space = cats_pca.transform(data_matrix)

plt.figure(figsize=(10, 6))
plt.scatter(pca_space[:, 0], pca_space[:, 1], c='b', marker='o', label='Cats')
plt.scatter(pca_space[[np.argmax(pc_1), np.argmin(pc_1)], 0], pca_space[[np.argmax(pc_1), np.argmin(pc_1)], 1], c='r', marker='x', s=100, label='Extreme in 1st PCA')
plt.scatter(pca_space[[np.argmax(pc_2), np.argmin(pc_2)], 0], pca_space[[np.argmax(pc_2), np.argmin(pc_2)], 1], c='g', marker='s', s=100, label='Extreme in 2nd PCA')
plt.xlabel('1st PCA Coordinate')
plt.ylabel('2nd PCA Coordinate')
plt.title('PCA Space')
plt.legend()
plt.grid(True)
plt.show()

#%% Ex 18)
#%% Ex 19) + 20)
w = -40000

synth_cat = cat_mean + w * cats_pca.components_[0, :]
synth_cat_im = create_u_byte_image_from_vector(synth_cat, height, width, channels)

io.imshow(synth_cat_im)
plt.title('The first synthesized cat')
io.show()

#%% Ex 21)
w = 40000
w1 = -9000

synth_cat = cat_mean + w * cats_pca.components_[0, :] + w1 * cats_pca.components_[1, :]
synth_cat_im = create_u_byte_image_from_vector(synth_cat, height, width, channels)

io.imshow(synth_cat_im)
plt.title('Synthesized cat using PC1 and PC2')
io.show()

#%% Ex 22)
m = 1 # Principal component to inspect

synth_cat_plus = cat_mean + 3 * np.sqrt(cats_pca.explained_variance_[m]) * cats_pca.components_[m, :]
synth_cat_minus = cat_mean - 3 * np.sqrt(cats_pca.explained_variance_[m]) * cats_pca.components_[m, :]

synth_cat_plus_im = create_u_byte_image_from_vector(synth_cat_plus, height, width, channels)
synth_cat_min_im = create_u_byte_image_from_vector(synth_cat_minus, height, width, channels)

io.imshow(synth_cat_plus_im)
plt.title('synth_cat_plus_im')
io.show()
io.imshow(synth_cat_min_im)
plt.title('synth_cat_min_im')
io.show()

# PC1 - eye color
# PC2 - fur color

#%% Ex 23)
import random

n_components_to_use = 3
synth_cat = cat_mean
for idx in range(n_components_to_use):
    w = random.uniform(-1, 1) * 3 * np.sqrt(cats_pca.explained_variance_[idx])
    synth_cat = synth_cat + w * cats_pca.components_[idx, :]
    synth_cat_im = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    
    io.imshow(synth_cat_im)
    plt.title('synth_cat_min_im ' + str(idx))
    io.show()

#%% 24)
im_miss = io.imread("ex8-CatsCatsCats/data/MissingCatProcessed.jpg")
im_miss_flat = im_miss.flatten()
im_miss_flat = im_miss_flat.reshape(1, -1)
pca_coords = cats_pca.transform(im_miss_flat)
pca_coords = pca_coords.flatten()

#%% 25)




























