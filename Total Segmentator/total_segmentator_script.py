Here's how you can prepare the scripts based on your requirements. Note that the actual implementation of some steps, especially the cropping of images based on segmentation coordinates, would typically involve some form of scripting beyond simple bash commands, likely involving Python and libraries for handling NIfTI files (such as nibabel). This part is simplified in the explanation since the precise method can vary greatly depending on the data structure and desired outcome.

### 1. Bash script for applying TotalSegmentator on each image

# bash

#!/bin/bash

# Directory paths
input_images_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTr"
input_labels_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTr"
output_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/brain_segmentations"

# Loop through the image files in the directory
for image_path in $input_images_dir/*.nii.gz; do
  # Extract the unique identifier
  file_name=$(basename "$image_path")
  unique_id="${file_name:3:4}"

  # Define output paths
  output_image_path="${output_dir}/Tr_brain_${unique_id}.nii.gz"

  # Apply TotalSegmentator on image
  TotalSegmentator -i "$image_path" -o "$output_image_path" -ta total --roi_subset brain --device gpu --statistics

  # Assuming similar naming convention for labels
  label_path="${input_labels_dir}/Tr_${unique_id}_0000.nii.gz"
  
  # Apply TotalSegmentator on label
  TotalSegmentator -i "$label_path" -o "$output_image_path" -ta total --roi_subset brain --device gpu --statistics
done


### 2. Script for cropping images

# This step requires a custom script, potentially in Python, to read the segmentation masks, extract the bounding box (range of all 3 coordinates) of the brain region, and then crop the original images and labels accordingly. This script is not detailed here due to the complexity and variation in implementation details.
import os
import nibabel as nib
import numpy as np

input_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTr'
output_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset058_IA/imagesTr'
mask_dir = 'data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/brain_segmentations'

for filename in os.listdir(input_dir):
    if filename.endswith('.nii.gz'):
        image = nib.load(os.path.join(input_dir, filename))
        mask = nib.load(os.path.join(mask_dir, 'Tr_brain_' + filename[3:7] + '.nii.gz'))

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        # Get the coordinates of the brain region
        coords = np.where(mask_data)

        # Crop the image to the brain region
        cropped_image_data = image_data[coords]

        # Create a new Nifti1Image with the cropped data
        cropped_image = nib.Nifti1Image(cropped_image_data, image.affine, image.header)

        # Save the cropped image
        nib.save(cropped_image, os.path.join(output_dir, filename))


### 3. Bash script for training the newly cropped images using nnUNet

#!/bin/bash

# Environment variables
export nnUNet_raw="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"  
export nnUNet_results="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"
export nnUNet_preprocessed="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"

# Preprocessing
nnUNetv2_plan_and_preprocess -d 058 --verify_dataset_integrity

# Training
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 058 3d_fullres 0 --npz

echo "It's time to run the prediction-post-processing-evaluation pipeline for 058 !"
