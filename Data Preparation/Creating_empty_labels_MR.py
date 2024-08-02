# First draft : nnunet format renaming 

import os
import shutil

# Define paths
src_dir = '/path/to/original/dataset'
dest_dir = '/path/to/new/dataset'
images_tr_dir = os.path.join(dest_dir, 'imagesTr')
labels_tr_dir = os.path.join(dest_dir, 'labelsTr')

# Create the destination directory and subdirectories
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)

# Function to determine the new file name based on old file name
def get_new_filename(old_filename):
    parts = old_filename.split('_')
    patient_id = parts[1]
    if 'Lesion' in old_filename:
        # This is a segmentation label file
        new_filename = f"{patient_id}_0000.nii.gz"
        target_dir = labels_tr_dir
    else:
        # This is an original brain image
        new_filename = f"{patient_id}_0000.nii.gz"
        target_dir = images_tr_dir
    return os.path.join(target_dir, new_filename)

# Copy and rename files
for filename in os.listdir(src_dir):
    src_file_path = os.path.join(src_dir, filename)
    dest_file_path = get_new_filename(filename)
    shutil.copy2(src_file_path, dest_file_path)

print("Files copied and renamed successfully.")


# Create the labelsTr directory if it does not exist
os.makedirs(dest_dir, exist_ok=True)

# Collect all patient IDs and check for lesion files
patient_ids = {}
for filename in os.listdir(src_dir):
    if 'desc-brain_mask.nii.gz' in filename:
        patient_id = filename.split('_desc')[0]
        patient_ids[patient_id] = patient_ids.get(patient_id, {'brain': None, 'Lesion': False})
        patient_ids[patient_id]['brain'] = filename
    if 'Lesion' in filename:
        lesion_patient_id = filename.split('_desc')[0]
        if lesion_patient_id in patient_ids:
            patient_ids[lesion_patient_id]['Lesion'] = True

# Create empty masks for control participants (those without lesions)
for patient_id, files in patient_ids.items():
    if not files['Lesion'] and files['brain']:
        src_file_path = os.path.join(src_dir, files['brain'])
        output_file_path = os.path.join(dest_dir, f"{patient_id}_0000.nii.gz")
        create_empty_mask(src_file_path, output_file_path)

print("Empty masks created for control participants.")


######################## 


# Creating empty masks 

# Careful, copy the dataset and test the code first 
# Test controls id from meta data and the generated empty labels 

import nibabel as nib
import numpy as np
import os
from nibabel.filebasedimages import ImageFileError
from tqdm import tqdm

def create_and_check_empty_mask(src_dir, dest_dir):
    filenames = os.listdir(src_dir)
    for filename in tqdm(filenames, desc="Processing images", total = len(filenames)):
        # Create the full path to the source image
        image_path = os.path.join(src_dir, filename)
        
        try:
            # Load the original image
            original_img = nib.load(image_path)
        except ImageFileError:
            print(f"Skipping file {filename} due to ImageFileError")
            continue
        
        # Create an empty mask with the same shape as the original image but all zeros
        empty_mask_data = np.zeros(original_img.shape, dtype=original_img.get_data_dtype())
        
        # Create a new NIfTI image with the empty mask, using the original image's header
        empty_mask_img = nib.Nifti1Image(empty_mask_data, original_img.affine, original_img.header)
        
        # Create the full path to the output image
        output_path = os.path.join(dest_dir, filename)
        
        # Save the empty mask
        nib.save(empty_mask_img, output_path)
        
        # Load the saved empty mask
        saved_empty_mask_img = nib.load(output_path)
        
        # Check if the header of the original image is the same as the header of the saved empty mask
        if original_img.header != saved_empty_mask_img.header:
            print(f"Header mismatch for file {filename}")

# Define the directory containing the participant images
src_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr'
dest_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr'

# Call the function
create_and_check_empty_mask(src_dir, dest_dir)


################### For one image 

import nibabel as nib
import numpy as np
import os
from nibabel.filebasedimages import ImageFileError

def create_and_check_empty_mask(image_path, output_path):
    try:
        # Load the original image
        original_img = nib.load(image_path)
    except ImageFileError:
        print(f"Skipping file {os.path.basename(image_path)} due to ImageFileError")
        return

    # Create an empty mask with the same shape as the original image but all zeros
    empty_mask_data = np.zeros(original_img.shape, dtype=original_img.get_data_dtype())

    # Create a new NIfTI image with the empty mask, using the original image's header
    empty_mask_img = nib.Nifti1Image(empty_mask_data, original_img.affine, original_img.header)

    # Save the empty mask
    nib.save(empty_mask_img, output_path)

    # Load the saved empty mask
    saved_empty_mask_img = nib.load(output_path)

    # Check if the header of the original image is the same as the header of the saved empty mask
    if original_img.header != saved_empty_mask_img.header:
        print(f"Header mismatch for file {os.path.basename(image_path)}")

# Define the path to the source image and the output path
image_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr/Tr_257_0000.nii.gz'
output_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr/Tr_257.nii.gz'

# Call the function
create_and_check_empty_mask(image_path, output_path)