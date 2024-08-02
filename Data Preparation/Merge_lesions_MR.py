# I need to merge multiple lesions into a single label file. 
# Files are named like this 
# sub-XXX_ses-xxxxxxxx_desc-Lesion_1_mask.nii.gz
# sub-XXX_ses-xxxxxxxx_desc-Lesion_2_mask.nii.gz
# number of lesions can go up to 4 
# the first XXX is the patient ID
# the next xxxxxxxx is the session date 

# I need to merge all the lesions from the same patient into a single label file

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm 

def merge_lesions(directory, output_directory):
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz') and 'Lesion' in f]
    grouped_files = {}

    # Group files by patient ID only
    for file in files:
        parts = file.split('_')
        patient_key = parts[0]  # patient_id only
        if patient_key not in grouped_files:
            grouped_files[patient_key] = []
        grouped_files[patient_key].append(file)

    # Process each group
    for key, filenames in tqdm(grouped_files.items(), desc="Merging lables", total = len(grouped_files)):
        merged_data = None
        previous_filename = None

        for filename in filenames:
            file_path = os.path.join(directory, filename)
            img = nib.load(file_path)
            data = img.get_fdata()

            if merged_data is None:
                merged_data = data
            else:
                try:
                    merged_data = np.logical_or(merged_data, data)
                except ValueError as e:
                    print(f"Error merging files {previous_filename} and {filename}: {e}")
                    continue

            previous_filename = filename

        # Save the merged data back as a new NIfTI file
        merged_img = nib.Nifti1Image(merged_data.astype(np.int16), img.affine, img.header)
        output_filename = f"{key}.nii.gz"
        nib.save(merged_img, os.path.join(output_directory, output_filename))
        print(f"Merged file saved as: {output_filename}")

# Specify the directory containing your .nii.gz files
#merge_lesions('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels', output_directory='/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr')

merge_lesions('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr', output_directory='/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr')



# Extract patients id that have sub_XXX_Lesion in their names (had merged lesions)

source_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr' 

import os

def extract_patient_ids(directory):
    # List all files in the directory
    filenames = os.listdir(directory)

    # Filter the list to include only files that end with '_Lesion.nii.gz'
    lesion_filenames = [f for f in filenames if f.endswith('_Lesion.nii.gz')]

    # Extract the patient IDs from the filenames
    patient_ids = [f.split('_')[0] for f in lesion_filenames]

    return patient_ids

# Specify the directory containing your .nii.gz files
source_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr'

# Call the function and print the patient IDs
patient_ids = extract_patient_ids(source_path)
for patient_id in patient_ids:
    print(patient_id)



################### Merge lesions for 1 single patient 

def merge_lesions(file_paths, output_directory):
    grouped_files = {}

    # Group files by patient ID only
    for file_path in file_paths:
        file = os.path.basename(file_path)
        parts = file.split('_')
        patient_key = '_'.join(parts[:2])  # patient_id consists of the first two parts
        if patient_key not in grouped_files:
            grouped_files[patient_key] = []
        grouped_files[patient_key].append(file_path)

    # Process each group
    for key, filepaths in tqdm(grouped_files.items(), desc="Merging lables", total = len(grouped_files)):
        merged_data = None
        previous_filepath = None

        for filepath in filepaths:
            img = nib.load(filepath)
            data = img.get_fdata()

            if merged_data is None:
                merged_data = data
            else:
                try:
                    merged_data = np.logical_or(merged_data, data)
                except ValueError as e:
                    print(f"Error merging files {previous_filepath} and {filepath}: {e}")
                    continue

            previous_filepath = filepath

        # Save the merged data back as a new NIfTI file
        merged_img = nib.Nifti1Image(merged_data.astype(np.int16), img.affine, img.header)
        output_filename = f"Tr_{key}_Lesion.nii.gz"
        nib.save(merged_img, os.path.join(output_directory, output_filename))
        print(f"Merged file saved as: {output_filename}")

# Specify the list of your .nii.gz files
merge_lesions([
    '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_1_mask.nii.gz',
    '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_2_mask.nii.gz',
    '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_3_mask.nii.gz'
], output_directory='/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr')


'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_1_mask.nii.gz'
'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_2_mask.nii.gz'
'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_3_mask.nii.gz'


def print_shape(image_path):
    """Load data from image, read nifti header and prints image shape"""
    img = nib.load(image_path)
    print(f"Shape for {image_path}: {img.get_fdata().shape}")

# Call the function for each image
print_shape('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_1_mask.nii.gz')
print_shape('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_2_mask.nii.gz')
print_shape('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-Lesion_3_mask.nii.gz')
print_shape('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels/manual_masks/sub-327/ses-20110923/anat/sub-327_ses-20110923_desc-brain_mask.nii.gz')
print_shape('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr/Tr_sub-327_ses-20110923_Lesion.nii.gz')
print_shape('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr/Tr_327_0000.nii.gz')