# This script allowed me to visualize the predicted aneurysms characteristics in a csv file. 

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import cyminiball
import pandas as pd
from tqdm import tqdm 

def statistics_table_folder(directory_path):
    # Initialize lists to store the aneurysm characteristics
    file_names = []
    aneurysm_ids = []
    sizes_voxels = []
    sizes_mm = []
    bounding_box_diagonals = []
    centers_of_mass = []
    # Get a list of all NIfTI files in the directory
    nifti_files = [file_name for file_name in os.listdir(directory_path) if file_name.endswith('.nii.gz')]

    # Iterate over all NIfTI files in the directory
    for file_name in tqdm(nifti_files, desc='Calculating statistics', total = len(nifti_files)):
        # Check if the file is a NIfTI file
        if file_name.endswith('.nii.gz'):
            # Load the NIfTI image
            img = nib.load(os.path.join(directory_path, file_name))

            # Get the image data as a numpy array
            mask = img.get_fdata()

            # Get the voxel size
            voxel_size = img.header.get_zooms()

            # Identify the connected components
            labels, num_labels = ndimage.label(mask)

             # If there are no labels, add a row with np.nan values
            if num_labels == 0:
                file_names.append(file_name)
                aneurysm_ids.append(np.nan)
                sizes_voxels.append(np.nan)
                sizes_mm.append(np.nan)
                bounding_box_diagonals.append(np.nan)
                centers_of_mass.append(np.nan)
            else:
                # Iterate over the connected components 
                for i in range(1, num_labels + 1):
                    # Get the coordinates of the voxels in the current component
                    coords = np.transpose(np.nonzero(labels == i)) 

                    # Compute the smallest enclosing ball of these coordinates
                    center, squared_radius = cyminiball.compute(coords)

                    # The radius is the square root of the squared radius
                    radius = np.sqrt(squared_radius)

                    # Convert the radius from voxels to millimeters
                    radius_mm = radius * voxel_size[0]  # assuming the voxel size is the same in all dimensions

                    # Compute the size as the diameter
                    size = 2 * radius_mm

                    # Compute the size in voxels
                    size_voxels = coords.shape[0]

                    # Compute the bounding box diagonal
                    bounding_box_diagonal = np.sqrt(np.sum((coords.max(axis=0) - coords.min(axis=0)) ** 2))

                    # Compute the center of mass
                    center_of_mass = ndimage.center_of_mass(labels == i)

                    # Store the aneurysm characteristics
                    file_names.append(file_name)
                    aneurysm_ids.append(i)
                    sizes_voxels.append(size_voxels)
                    sizes_mm.append(size)
                    bounding_box_diagonals.append(bounding_box_diagonal)
                    centers_of_mass.append(center_of_mass)

    # Create a DataFrame
    df = pd.DataFrame({
        'file_name': file_names,
        'aneurysm_id': aneurysm_ids,
        'size_voxels': sizes_voxels,
        'size_mm': sizes_mm,
        'bounding_box_diagonal': bounding_box_diagonals,
        'center_of_mass': centers_of_mass
    })

    # Save the DataFrame as a CSV file
    df.to_csv('statistics_internal_2000_f0.csv', index=False)

    # Print the DataFrame
    print(df.head())

# Call the function
statistics_table_folder('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_external/')
# Create table for postprocessed internal test set (predicted) 
statistics_table_folder('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_all_folds')
# Create table for postprocessed internal test set (predicted 2000 epochs) 
statistics_table_folder('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_2000_f0')
# Create table for training set (ground truth)
# Change the name of the file to statistics_table_train_ground_truth.csv ! 
statistics_table_folder('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTr/')



# Add radius column to the table 

def add_radius_column(csv_file, output_file):
    # Load the data
    df = pd.read_csv(csv_file)

    # Compute the radius and add it as a new column
    df['radius_voxels'] = df['size_voxels'] / 2

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)


add_radius_column('stats_internal_gt.csv', 'stats_internal_gt.csv')
add_radius_column('stats_internal_predicted.csv', 'stats_internal_predicted.csv')
add_radius_column('statistics_internal_2000_f0.csv', 'stats_internal_2000_f0.csv')


# Convert center of mass from string to tuples 

# Load the data
df = pd.read_csv('stats_internal_gt.csv')
# Function to convert a string to a tuple of floats, handling 'nan' values
def str_to_tuple(s):
    if pd.isna(s):
        return np.nan
    else:
        return tuple(map(float, s.strip('()').split(',')))
# Convert the 'center_of_mass' column from strings to tuples
df['center_of_mass'] = df['center_of_mass'].apply(str_to_tuple)
# Save the DataFrame to a new CSV file
df.to_csv('stats_internal_gt_converted.csv', index=False)


# Filter the DataFrame to find the row with size_mm == 80
outlier = df[df['size_mm'] == 80]
# Print the characteristics of the outlier
print(outlier)


# Visualize statistics 

import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the DataFrame from the CSV file
df = pd.read_csv('statistics_table_train_gt.csv')

# Remove aneurysms bigger than 20 mm from the DataFrame
df = df[df['size_mm'] <= 20]

# Convert the 'center_of_mass' column back to tuples
df['center_of_mass'] = df['center_of_mass'].apply(ast.literal_eval)

# Create a histogram of the aneurysm sizes
plt.figure(figsize=(10, 6))
plt.hist(df['size_mm'], bins=30, edgecolor='black')
plt.title('Histogram of Aneurysm Sizes')
plt.xlabel('Size (mm)')
plt.ylabel('Frequency')
plt.show()

# Create a histogram of the bounding box diagonals
plt.figure(figsize=(10, 6))
plt.hist(df['bounding_box_diagonal'], bins=30, edgecolor='black')
plt.title('Histogram of Bounding Box Diagonals')
plt.xlabel('Bounding Box Diagonal')
plt.ylabel('Frequency')
plt.show()

