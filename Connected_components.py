# Norman's code to compute connected components and identify separate aneurysms after segmentation

import scipy.ndimage as ndimage
import numpy as np

mask = np.zeros((100,100,100), dtype=bool)
print(mask.shape)

# Blob 1
mask[10:20,10:20,10:20] = True
# Blob 2
mask[50:200,40:90,10:20] = True
# Blob 3
mask[60:70,10:30, 50:90] = True

# Identify the connected components
labels, n_labels = ndimage.label(mask)

# Compute average lesion size
volumes = [np.sum(labels == l) for l in np.unique(labels) if l !=0 ]

# Print some output
print(n_labels)
print("Blob 1: ", np.sum(labels==1))  # Number of voxels in blob 1
print("Blob 2: ", np.sum(labels==2))  # Number of voxels in blob 2
print("Blob 3: ", np.sum(labels==3))  # Number of voxels in blob 3
print("Test:   ", np.sum(labels>0) == np.sum(mask))  # Should be true
print("Volumes:", volumes)
print("Mean:   ", "%.2f" % np.mean(volumes))

# Take into account voxel dimensions to determine the volume!


# Trying to read the label of a image with 3 aneurysms as an example to check the code 

import nibabel as nib
import numpy as np

# Load the NIfTI image
img = nib.load('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal/Ts_0021.nii.gz')

# Get the image data as a numpy array
img_data = img.get_fdata()

# Find the indices where the image data is non-zero (True)
true_indices = np.nonzero(img_data)

print(true_indices)

# Identify the connected components
labels, n_labels = ndimage.label(img_data)

print(n_labels)

# Now let's compare with what nnunet predicted 

# Load the NIfTI image
img = nib.load('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/predicted_internal_f0/Ts_0021.nii.gz')

# Get the image data as a numpy array
img_data = img.get_fdata()

# Find the indices where the image data is non-zero (True)
true_indices = np.nonzero(img_data)

print(true_indices)

# Identify the connected components
labels, n_labels = ndimage.label(img_data)

print(n_labels)

# Also found 3 aneurysms 

# Now let's try to compute the volume of each aneurysm/ mask size 

import numpy as np
from scipy import ndimage

def size_mask(mask):
    labels, num_labels = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    print(sizes)
    return sizes

size_mask(img_data)

# Final function 
import os
import nibabel as nib
import numpy as np
from scipy import ndimage

def compute_components_and_sizes(image_path):
    # Load the NIfTI image
    img = nib.load(image_path)
    # Get the base name of the file
    base_name = os.path.basename(image_path)


    # Get the image data as a numpy array
    img_data = img.get_fdata()

    # Identify the connected components
    labels, n_labels = ndimage.label(img_data)

    print(f'Number of connected components in {base_name}: {n_labels}')

    # Compute the size of each connected component
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    print(f'Sizes in voxels: {sizes}')

    return n_labels, sizes, labels

# Use the function
n_labels, sizes, labels = compute_components_and_sizes('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal/Ts_0021.nii.gz')


# Loop through all the files in the directory to apply the function to all the masks

import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import pandas as pd

folder_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal'

def compute__all_components_and_sizes(folder_path):
    """ Take the folder of masks, iterate through  them, 
    calculate the number of connected components and 
    the size of each component
    Save in a csv table"""
    # List all the files in the folder
    files = os.listdir(folder_path)
    # Initialize lists to store the results
    base_names_list = []
    n_labels_list = []
    sizes_list = []
    # Iterate through the files
    for file in files:
        # Compute the components and sizes
        n_labels, sizes = compute_components_and_sizes(os.path.join(folder_path, file))
        # Store the results
        base_names_list.append(os.path.basename(file))
        n_labels_list.append(n_labels)
        sizes_list.append(sizes)
    
    # Save in a csv table
    df = pd.DataFrame({
        'file': base_names_list,
        'n_labels': n_labels_list,
        'sizes': sizes_list
    })
    df.to_csv('components_and_sizes_gt_masks_internal_057.csv', index=False)

    return n_labels_list, sizes_list

# Use the function
n_labels_list, sizes_list = compute__all_components_and_sizes(folder_path)