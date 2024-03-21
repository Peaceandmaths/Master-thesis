# I'm trying the total_segmentator_script.py in Python to see if it works on 10 images. 

### 1. Bash script for applying TotalSegmentator on each image

# bash script 

#!/bin/bash
#!/bin/bash

# Directory paths
input_images_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTr"
input_labels_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/labelsTr"
output_images_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_brain/imagesTr"
output_labels_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_brain/labelsTr"

# Get the total number of files
total_files=$(ls $input_images_dir/*.nii.gz | wc -l)

# Loop through the image files in the directory
for image_path in $input_images_dir/*.nii.gz; do
  # Extract the file name
  file_name=$(basename "$image_path")
  
  # Assuming the unique_id is part of the file name, extract it correctly
  # Here you need to modify the extraction method according to your file naming convention
  # For example, if file names are in the format "Tr_uniqueID_0000.nii.gz", you can extract the unique_id like this:
  unique_id=$(echo $file_name | sed 's/Tr_\(.*\)_0000.nii.gz/\1/')
  
  # If the unique_id extraction logic above is incorrect due to a different file naming convention,
  # adjust the 'sed' command accordingly to match your files' naming pattern

  # Define output paths for images and labels using the file name directly (not unique_id), 
  # since TotalSegmentator processes the whole file
  output_image_path="${output_images_dir}/Tr_brain_${unique_id}_0000.nii.gz"
  output_label_path="${output_labels_dir}/Tr_brain_${unique_id}.nii.gz"

   # Apply TotalSegmentator on image
  TotalSegmentator -i "$image_path" -o "$output_image_path" -ta total --roi_subset brain --device gpu --statistics
  # Check if TotalSegmentator succeeded
    if [ $? -ne 0 ]; then
        echo "TotalSegmentator failed for $image_path"
        exit 1
    else
        echo "TotalSegmentator finished successfully for $image_path"
    fi

  # Construct the label path using the unique_id extracted from the image file name
  label_path="${input_labels_dir}/Tr_${unique_id}.nii.gz"
  
  # Apply TotalSegmentator on label
  TotalSegmentator -i "$label_path" -o "$output_label_path" -ta total --roi_subset brain --device gpu --statistics
  # Check if TotalSegmentator succeeded
    if [ $? -ne 0 ]; then
        echo "TotalSegmentator failed for $label_path"
        exit 1
    else
        echo "TotalSegmentator finished successfully for $label_path"
    fi

done

### 2. Python Script for cropping images

# This step requires a custom script, potentially in Python, to read the segmentation masks, 
# extract the bounding box (range of all 3 coordinates) of the brain region, 
# and then crop the original images and labels accordingly. 

import os
import nibabel as nib

import numpy as np


input_im_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTr'
output_im_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_cropped/imagesTr'
mask_im_dir = 'data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_brain/imagesTr'

input_lab_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/labelsTr'
output_lab_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_cropped/labelsTr'
mask_lab_dir = 'data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_brain/labelsTr'

for filename in os.listdir(input_im_dir):
    if filename.endswith('.nii.gz'):
        unique_id = filename[3:7]  # Extract the unique identifier

        # Process the image
        image = nib.load(os.path.join(input_im_dir,  'Tr_' + unique_id + '_0000.nii.gz'))
        mask = nib.load(os.path.join(mask_im_dir, 'Tr_brain_' + unique_id + '.nii.gz'))

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        # Set all voxels outside of the mask to 0
        image_data[mask_data == 0] = 0

        # Get the coordinates of the brain region
        coords = np.where(mask_data)

        # Crop the image to the brain region
        cropped_image_data = image_data[coords]

        # Create a new Nifti1Image with the cropped data
        cropped_image = nib.Nifti1Image(cropped_image_data, image.affine, image.header)

        # Save the cropped image
        nib.save(cropped_image, os.path.join(output_im_dir, filename))

        # Process the label
        label = nib.load(os.path.join(input_lab_dir, 'Tr_' + unique_id + '.nii.gz'))
        mask = nib.load(os.path.join(mask_lab_dir, 'Tr_brain_' + unique_id + '.nii.gz'))

        label_data = label.get_fdata()
        mask_data = mask.get_fdata()

        # Set all voxels outside of the mask to 0
        label_data[mask_data == 0] = 0

        # Crop the label to the brain region
        cropped_label_data = label_data[coords]

        # Create a new Nifti1Image with the cropped data
        cropped_label = nib.Nifti1Image(cropped_label_data, label.affine, label.header)

        # Save the cropped label
        nib.save(cropped_label, os.path.join(output_lab_dir, 'Tr_cropped' + unique_id + '.nii.gz'))


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

