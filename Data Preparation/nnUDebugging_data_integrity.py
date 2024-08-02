
#Checking for data format compatibility 
import os
import json

# Replace with the path to your dataset folder
dataset_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset230_IA/'

# Check if the required files and folders exist
assert os.path.isfile(os.path.join(dataset_folder, 'dataset.json')), 'dataset.json not found'
assert os.path.isdir(os.path.join(dataset_folder, 'imagesTr')), 'imagesTr folder not found'
assert os.path.isdir(os.path.join(dataset_folder, 'labelsTr')), 'labelsTr folder not found'

# Load the dataset.json file
with open(os.path.join(dataset_folder, 'dataset.json')) as f:
    dataset_json = json.load(f)

# Check if the required keys are in the dataset.json file
required_keys = ['labels', 'channel_names', 'numTraining', 'file_ending']
for key in required_keys:
    assert key in dataset_json, f'{key} not found in dataset.json'

print('Dataset format looks correct')


# Checking for correct names and extensions
""" 
import os
import json

# Replace with the path to your dataset folder
dataset_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset230_IA/'

# Load the dataset.json file
with open(os.path.join(dataset_folder, 'dataset.json')) as f:
    dataset_json = json.load(f)

# Get the expected file extension from the dataset.json file
expected_extension = dataset_json['file_ending']

# Get the list of file names in the imagesTr and labelsTr folders
image_files = os.listdir(os.path.join(dataset_folder, 'imagesTr'))
label_files = os.listdir(os.path.join(dataset_folder, 'labelsTr'))

# Check if the file names have the expected extension
for file_name in image_files + label_files:
    if not file_name.endswith(expected_extension):
        print('Unexpected file extension:', file_name) """


# Checking for corrupted files
""" 
import os
import nibabel as nib

# Replace with the path to your dataset folder
dataset_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset230_IA/'

# Get the list of file paths in the imagesTr and labelsTr folders
image_files = [os.path.join(dataset_folder, 'imagesTr', f) for f in os.listdir(os.path.join(dataset_folder, 'imagesTr'))]
label_files = [os.path.join(dataset_folder, 'labelsTr', f) for f in os.listdir(os.path.join(dataset_folder, 'labelsTr'))]

# Try to open each file
for file_path in image_files + label_files:
    try:
        nib.load(file_path)
    except Exception as e:
        print(f'Error reading file {file_path}: {e}') """


# Renaming files 

# Debug it doens't change anything 

import os

def rename_files_in_subfolders(parent_folder):
    '''For files with a prefix Ts in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTr', 'labelsTr']

    for subfolder in subfolders:
        unique_id = 1887  # Reset the unique_id for each subfolder
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)
        for file in files:
            if file.startswith('Ts'):
                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{unique_id}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{unique_id}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Increment the unique identifier
                unique_id += 1
            elif file.startswith('Tr') and not file.endswith('_0000.nii.gz'):
                # Extract the unique identifier
                unique_id = file[2:6]  # Assumes the unique identifier is 4 digits

                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{unique_id}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{unique_id}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

rename_files_in_subfolders('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset002_IA/')