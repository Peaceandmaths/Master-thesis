# Raw data copy is in the Dataset001_IA folder
# The whole traianing + internal_validation set is = 1186 + 152 = 1338 files

# Code to copy data into another folder (in the terminal)

# cp -a /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/  /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset003_IA/

# Renaming files 
import os

def rename_files_in_subfolders(parent_folder):
    '''For files with a prefix Ts in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTr', 'labelsTr']

    for subfolder in subfolders:
        unique_id = 1187  # Reset the unique_id for each subfolder
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)
        for file in files:
            if file.startswith('Ts'):
                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Increment the unique identifier
                unique_id += 1
            elif file.startswith('Tr') and not file.endswith('_0000.nii.gz'):
                # Extract the unique identifier and convert it to an integer
                unique_id = int(file[2:6])  # Assumes the unique identifier is 4 digits

                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

rename_files_in_subfolders('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset003_IA/')

# Activate the environmnet master_thesis where nnunet is installed

# cd /home/golubeka/
# source master_thesis/bin/activate


# Run setting env variables in terminal before executing nnunet commands 

# export nnUNet_raw="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"  
# export nnUNet_results="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"
# export nnUNet_preprocessed="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"

# Verify data integrtity and create plans and populate preprocessing folder (in the terminal)

# nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity

# Run training (in the terminal)
# To see additional options add -h flag 
# nnUNetv2_train -h 

# Specify GPUs and run training on Fold 0
# CUDA_VISIBLE_DEVICES=0,2,5 nnUNetv2_train 002 3d_fullres 0
