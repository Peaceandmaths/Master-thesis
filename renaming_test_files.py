''' Renaming test images to nnunet format,
Implement a lookup table to remember the names, 
ensuring the images and labels are renamed consistently'''


# Fisrt, copy the original test dataset to another folder 
import shutil
import os

# Define source and destination directories
src_dir_images = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal'
src_dir_labels = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal'
dest_dir_images = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs'
dest_dir_labels = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs'


def copy_test_dataset(src_dir_images, src_dir_labels, dest_dir_images, dest_dir_labels):
    # Get list of files
    files_images = os.listdir(src_dir_images)
    files_labels = os.listdir(src_dir_labels)

    # Ensure that for every image file, there is a corresponding label file
    for file in files_images:
        if file not in files_labels:
            raise ValueError(f"No matching label for image {file}")
        
    # Make sure destination directories exist
    os.makedirs(dest_dir_images, exist_ok=True)
    os.makedirs(dest_dir_labels, exist_ok=True)

    # Copy selected files
    for file in files_images:
        shutil.copy(os.path.join(src_dir_images, file), dest_dir_images)

    for file in files_labels:
        shutil.copy(os.path.join(src_dir_labels, file), dest_dir_labels)


copy_test_dataset(src_dir_images,src_dir_labels, dest_dir_images, dest_dir_labels)
# Renaming with lookup table
import pandas as pd 
import os 

def rename_ext_test_nnunet_format(parent_folder):
    '''For files with a prefix ExtA and ExtB in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTs_external', 'labelsTs_external']
    lookup_table = {}

    for subfolder in subfolders:
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)
        # Count the number of files that start with 'ExtA'
        num_tr_files = sum([1 for file in files if file.startswith('ExtA')])

        # Start the unique_id at num_tr_files + 1
        unique_id = num_tr_files + 1

        # Keep track of used unique IDs
        used_ids = set()

        for file in files:
            if file.startswith('ExtB') or file.startswith('ExtA'):
                # Ensure the unique ID is unique
                while unique_id in used_ids:
                    unique_id += 1

                # Construct the new file name
                if subfolder == 'imagesTs_external':
                    new_file_name = f'Ts_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTs'
                    new_file_name = f'Ts_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Add the old and new names to the lookup table
                lookup_table[file] = new_file_name

                # Add the unique ID to the set of used IDs
                used_ids.add(unique_id)

                # Increment the unique identifier
                unique_id += 1
            elif file.startswith('Ts') and not file.endswith('_0000.nii.gz'):
                # Extract the unique identifier and convert it to an integer
                unique_id = int(file[2:6])  # Assumes the unique identifier is 4 digits

                # Ensure the unique ID is unique
                while unique_id in used_ids:
                    unique_id += 1

                # Construct the new file name
                if subfolder == 'imagesTs_external':
                    new_file_name = f'Ts_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Ts_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Add the old and new names to the lookup table
                lookup_table[file] = new_file_name

                # Add the unique ID to the set of used IDs
                used_ids.add(unique_id)

    # Convert the lookup table to a pandas DataFrame and save it to a CSV file
    df = pd.DataFrame(list(lookup_table.items()), columns=['Old Name', 'New Name'])
    df.to_csv('lookup_external_test_057.csv', index=False)

parent_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/'
rename_ext_test_nnunet_format(parent_folder)


# Renaming internal test set with lookup table 

import pandas as pd 
import os 

def rename_int_test_files_nnunet_format(parent_folder):
    '''For files with a prefix Ts in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTs_internal', 'labelsTs_internal']
    lookup_table = {}

    for subfolder in subfolders:
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)

        unique_id = 1
        used_ids = set()

        for file in files:
            if file.startswith('Ts') and not file.endswith('_0000.nii.gz'):
                while unique_id in used_ids:
                    unique_id += 1

                if subfolder == 'imagesTs_internal':
                    new_file_name = f'Ts_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTs_external'
                    new_file_name = f'Ts_{str(unique_id).zfill(4)}.nii.gz'

                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))
                lookup_table[file] = new_file_name
                used_ids.add(unique_id)
                unique_id += 1

    df = pd.DataFrame(list(lookup_table.items()), columns=['Old Name', 'New Name'])
    df.to_csv('lookup_internal_test_057.csv', index=False)

parent_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/'
rename_int_test_files_nnunet_format(parent_folder)

# Renaming internal test set with lookup table insuring images and labels are renamed consistently

import pandas as pd 
import os 

def rename_test_consistent(parent_folder):
    '''For files with a prefix Ts in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTs', 'labelsTs']
    lookup_table = {}

    # Get the sorted lists of image and label files
    image_files = sorted(os.listdir(os.path.join(parent_folder, 'imagesTs')))
    label_files = sorted(os.listdir(os.path.join(parent_folder, 'labelsTs')))

    unique_id = 1
    used_ids = set()

    # Iterate over the image and label files simultaneously
    for image_file, label_file in zip(image_files, label_files):
        if image_file.startswith('Ts') and not image_file.endswith('_0000.nii.gz'):
            while unique_id in used_ids:
                unique_id += 1

            # Rename the image file
            new_image_name = f'Ts_{str(unique_id).zfill(4)}_0000.nii.gz'
            os.rename(os.path.join(parent_folder, 'imagesTs', image_file), os.path.join(parent_folder, 'imagesTs', new_image_name))
            lookup_table[image_file] = new_image_name

            # Rename the label file
            new_label_name = f'Ts_{str(unique_id).zfill(4)}.nii.gz'
            os.rename(os.path.join(parent_folder, 'labelsTs', label_file), os.path.join(parent_folder, 'labelsTs', new_label_name))
            lookup_table[label_file] = new_label_name

            used_ids.add(unique_id)
            unique_id += 1

    df = pd.DataFrame(list(lookup_table.items()), columns=['Old Name', 'New Name'])
    df.to_csv('lookup_test.csv', index=False)

parent_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/'
rename_test_files_nnunet_format(parent_folder)