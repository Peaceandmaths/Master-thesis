import os
import nibabel as nib

def list_low_resolution_nifti_images(directory, threshold=(0.7, 0.7, 0.7)):
    """
    List NIfTI images with resolution lower than the specified threshold.

    Parameters:
    - directory: Path to the directory containing NIfTI files.
    - threshold: A tuple of voxel sizes (x, y, z) considered as the threshold for low resolution.

    Returns:
    - A list of filenames of NIfTI images considered low resolution.
    """
    low_res_images = []
    for filename in os.listdir(directory):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            filepath = os.path.join(directory, filename)
            try:
                nifti_img = nib.load(filepath)
                header = nifti_img.header
                #print(header)
                voxel_sizes = header.get_zooms()[:3]  # get_zooms() to get the voxel sizes in millimeters:
                #print(voxel_sizes)

                # Compare voxel sizes to threshold
                if any(voxel_size > threshold_dim for voxel_size, threshold_dim in zip(voxel_sizes, threshold)):
                    low_res_images.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return low_res_images

# Example usage on 10 images (train) to see if code works
#directory = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTr' 
#low_res_images = list_low_resolution_nifti_images(directory)
#print("Low resolution NIfTI images:", low_res_images)

# Comparing with test images resolution (73 images small dataset)

directory = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTs_internal'  
low_res_images = list_low_resolution_nifti_images(directory)
print("Low resolution NIfTI images:", low_res_images)
print(len(low_res_images))

# Checking on small train and test sets (055) and storing in a table for comparison

import pandas as pd
import numpy as np

# Initialize an empty list to hold the data before creating a DataFrame
data = []

# List of directories to check
directories = ['/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTs_internal',
               '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTr']

# Thresholds for voxel sizes
thresholds = np.arange(0.1, 0.9, 0.1)

# Assuming list_low_resolution_nifti_images is correctly defined elsewhere in your code

# Loop through the directories
for directory in directories:
    # Extract the base name of the directory
    directory_name = os.path.basename(directory)

    # Loop through the thresholds
    for threshold in thresholds:
        # Get the low resolution images
        # IMPORTANT: Ensure list_low_resolution_nifti_images function is correctly defined to use the new threshold
        low_res_images = list_low_resolution_nifti_images(directory, (threshold, threshold, threshold))
        # Add the data to the list
        data.append({'Directory': directory_name, 'Threshold': threshold, 'Number of Images': len(low_res_images)})

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)



# Checking on big train and test sets (057) and storing in a table for comparison

import pandas as pd
import numpy as np

# Initialize an empty list to hold the data before creating a DataFrame
data = []

# List of directories to check
directories = ['/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal',
               '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTr']

# Thresholds for voxel sizes
thresholds = np.arange(0.3, 1.1, 0.1)

# Assuming list_low_resolution_nifti_images is correctly defined elsewhere in your code

# Loop through the directories
for directory in directories:
    # Extract the base name of the directory
    directory_name = os.path.basename(directory)

    # Loop through the thresholds
    for threshold in thresholds:
        # Get the low resolution images
        # IMPORTANT: Ensure list_low_resolution_nifti_images function is correctly defined to use the new threshold
        low_res_images = list_low_resolution_nifti_images(directory, (threshold, threshold, threshold))
        # Add the data to the list
        data.append({'Directory': directory_name, 'Threshold': threshold, 'Number of Images': len(low_res_images)})

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Pivot the DataFrame
df_pivot = df.pivot(index='Threshold', columns='Directory', values='Number of Images')

# Print the pivoted DataFrame
print(df_pivot)


# Checking  images in big set Tr that have size bigger than 0.9 

directory = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTr' 
low_res_images = list_low_resolution_nifti_images(directory)
print("Low resolution NIfTI images:", low_res_images)
