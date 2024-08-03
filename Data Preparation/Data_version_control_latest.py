""" 

This code described the data preparation workflow, from copying the raw data, creting small subset for testing, 
renaming files according to the nnunet format (added lookup table to track the change), 
to reorganizing files into nnunet folders and running nnnuent scripts ( verify data integrity, preprocessing, training, evaluation, post-processing).

There's a history of my experiments with changin the default number of epochs ( from 1000 to 2000) and comparing results from 5 folds with the best model automatically chosen by nnunet. 

Finally, it contains a function to create a bash script that runs the whole pipeline, given the dataset id and fold number. 

"""

# Raw data copy is in the Dataset001_IA folder
# The whole traianing + internal_validation set is = 1186 + 152 = 1338 files

# Code to copy data into another folder (in the terminal)

# cp -a /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/  /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset003_IA/

# Copy a small datatset for testing 10% of the original dataset

import os
import shutil
import random

# Define source and destination directories
src_dir_images = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/imagesTr'
src_dir_labels = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/labelsTr'
dest_dir_images = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/imagesTr'
dest_dir_labels = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/labelsTr'


def create_small_dataset(src_dir_images, src_dir_labels, dest_dir_images, dest_dir_labels):
    """ Select 108 files from the first 1186 files and 27 files after the 1186th file( that have Ts prefix)"""
    # Get list of files
    files_images = sorted(os.listdir(src_dir_images))
    files_labels = sorted(os.listdir(src_dir_labels))

    # Ensure that for every image file, there is a corresponding label file
    for file in files_images:
        if file not in files_labels:
            raise ValueError(f"No matching label for image {file}")

    # Select 108 files from the first 1186 files and 27 files after the 1186th file
    selected_files_images = files_images[:108] + files_images[1186:1186+27]
    selected_files_labels = [f for f in selected_files_images]

    # Make sure destination directories exist
    os.makedirs(dest_dir_images, exist_ok=True)
    os.makedirs(dest_dir_labels, exist_ok=True)

    # Copy selected files
    for file in selected_files_images:
        shutil.copy(os.path.join(src_dir_images, file), dest_dir_images)

    for file in selected_files_labels:
        shutil.copy(os.path.join(src_dir_labels, file), dest_dir_labels)

create_small_dataset(src_dir_images, src_dir_labels, dest_dir_images, dest_dir_labels)


# Renaming files 
import os

# This is an old version that worked
def rename_files_in_subfolders(parent_folder):
    '''For files with a prefix Ts in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTr', 'labelsTr']

    for subfolder in subfolders:
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)
        # Count the number of files that start with 'Tr'
        num_tr_files = sum([1 for file in files if file.startswith('Tr')])

        # Start the unique_id at num_tr_files + 1
        unique_id = num_tr_files + 1

        # Keep track of used unique IDs
        used_ids = set()

        for file in files:
            if file.startswith('Ts'):
                # Ensure the unique ID is unique
                while unique_id in used_ids:
                    unique_id += 1

                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Add the unique ID to the set of used IDs
                used_ids.add(unique_id)

                # Increment the unique identifier
                unique_id += 1
            elif file.startswith('Tr') and not file.endswith('_0000.nii.gz'):
                # Extract the unique identifier and convert it to an integer
                unique_id = int(file[2:6])  # Assumes the unique identifier is 4 digits

                # Ensure the unique ID is unique
                while unique_id in used_ids:
                    unique_id += 1

                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Add the unique ID to the set of used IDs
                used_ids.add(unique_id)
rename_files_in_subfolders('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/')

import pandas as pd
import os

# Renaming funciton that messes up uniqueids and saves in a lookup table
def rename_training_files(parent_folder):
    '''For files with a prefix Tr in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier'''
    subfolders = ['imagesTr', 'labelsTr']
    lookup_table = []

    for subfolder in subfolders:
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)

        # Start the unique_id at 1
        unique_id = 1

        # Keep track of used unique IDs
        used_ids = set()

        for file in files:
            if file.startswith('Tr') and not file.endswith('_0000.nii.gz'):
                # Ensure the unique ID is unique
                while unique_id in used_ids:
                    unique_id += 1

                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{str(unique_id).zfill(4)}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Add the unique ID to the set of used IDs
                used_ids.add(unique_id)

                # Add old and new file names to lookup table
                lookup_table.append((file, new_file_name))

                # Increment the unique identifier
                unique_id += 1

    # Convert lookup table to DataFrame and save to CSV
    lookup_table_df = pd.DataFrame(lookup_table, columns=['Old Name', 'New Name'])
    lookup_table_df.to_csv('lookup_table.csv', index=False)

rename_training_files('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/')

# Revert renaming 
def revert_renaming(parent_folder, lookup_table_csv):
    '''Revert the renaming of files based on a lookup table'''
    subfolders = ['imagesTr', 'labelsTr']

    # Load lookup table from CSV
    lookup_table_df = pd.read_csv(lookup_table_csv)

    for subfolder in subfolders:
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)

        for _, row in lookup_table_df.iterrows():
            old_name, new_name = row['Old Name'], row['New Name']
            if new_name in files:
                # Rename the file back to the old name
                os.rename(os.path.join(folder, new_name), os.path.join(folder, old_name))

revert_renaming('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/', 'lookup_table.csv')


# New function, still messes up the labels, but at least image-label is coherent 
def rename_files_in_subfolders(parent_folder):
    '''For files with a prefix Tr in a folder, change the ending to _XXXX_0000.nii.gz where XXXX is a unique identifier generated based on the file's position in the list of files'''
    subfolders = ['imagesTr', 'labelsTr']
    lookup_table = []

    for subfolder in subfolders:
        folder = os.path.join(parent_folder, subfolder)
        files = os.listdir(folder)

        for i, file in enumerate(files):
            if file.startswith('Tr') and not file.endswith('_0000.nii.gz'):
                # Use the file's position in the list of files as the unique identifier
                unique_id = str(i).zfill(4)

                # Construct the new file name
                if subfolder == 'imagesTr':
                    new_file_name = f'Tr_{unique_id}_0000.nii.gz'
                else:  # subfolder == 'labelsTr'
                    new_file_name = f'Tr_{unique_id}.nii.gz'

                # Rename the file
                os.rename(os.path.join(folder, file), os.path.join(folder, new_file_name))

                # Add old and new file names to lookup table
                lookup_table.append((file, new_file_name))

    # Convert lookup table to DataFrame and save to CSV
    lookup_table_df = pd.DataFrame(lookup_table, columns=['Old Name', 'New Name'])
    lookup_table_df.to_csv('lookup_table.csv', index=False)

rename_files_in_subfolders('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/')


# Renaming test dataset 
from renaming_test_files import rename_test_files_nnunet_format
parent_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/'
rename_test_files_nnunet_format(parent_folder)

# I renamed the external and internal test set with lookup table (but only for lables, not images)
# The corresponding images can be found with the old name + see lookup_table_Ts.csv
# If lost, reload the original test sets and rename it again with a lookup table for both images and labels 

from renaming_test_files import rename_test_consistent
parent_folder = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/'
rename_test_files_nnunet_format(parent_folder) # Modify the names of the folders and the lookup table name 

# Add dataset.json file and change the num of files in the json file


# Activate the environmnet master_thesis where nnunet is installed

# tmux at
# cd /home/golubeka/
# source master_thesis/bin/activate


# Run setting env variables in terminal before executing nnunet commands 

# export nnUNet_raw="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"  
# export nnUNet_results="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"
# export nnUNet_preprocessed="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"

# Verify data integrtity and create plans and populate preprocessing folder (in the terminal)

# Before running heavy commands, make sure to delete unnecessary files, make sure there's enough space on the disk 
# nnUNetv2_plan_and_preprocess -d 057 --verify_dataset_integrity

# Move files starting from Ts to a differnet folder 

import os
import shutil
def movefiles(src_dir, dest_dir):
    files = os.listdir(src_dir)
    for file in files:
        if file.startswith('Ts'):
            shutil.move(os.path.join(src_dir, file), dest_dir)

movefiles('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/imagesTr', '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/imagesTs_internal')
# same for labels 
movefiles('/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/labelsTr', '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/labelsTs_internal')

# After preprocessing is over, define splits manually by changing the splits_final.json file 

# No let nnunet define it's own 5 folds ! 
from manual_splits import define_splits_small
# from manual_splits import define_splits_whole
save_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed/Dataset050_IA'
define_splits_small(save_path)

# Run training (in the terminal)
# To see additional options add -h flag 
# nnUNetv2_train -h 

# Specify GPUs and run training on Fold 0

# export CUDA_VISIBLE_DEVICES=0,2,5
# CUDA_VISIBLE_DEVICES=0,2,5 nnUNetv2_train 005 3d_fullres 0
# Continue training from last checkpoint 
""" 
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 050 3d_fullres 0 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_latest.pth &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 050 3d_fullres 1 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_latest.pth &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 050 3d_fullres 2 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_latest.pth
 """


# CUDA_VISIBLE_DEVICES=0,2,5 nnUNetv2_train 005 3d_fullres 0 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset005_IA/

# Using multiple GPUs
""" 
Using GPUs 0,1,2 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 057 3d_fullres 0 --npz & 
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 057 3d_fullres 1 --npz & 
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 057 3d_fullres 2 --npz  

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 050 3d_fullres 3 --npz & 
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 050 3d_fullres 4 --npz 

 """

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 050 3d_fullres 0 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/ &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 050 3d_fullres 1 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/ &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 050 3d_fullres 2 -pretrained_weights /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/



# Using 1 GPU on 1 fold 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 057 3d_fullres 0 --npz 

# Kill a process 

# kill -TERM PID 
""" 
1) Find best configuration  

nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 

Only for certain folds 
nnUNetv2_find_best_configuration 050 -c 3d_fullres -f 0 1 2 3 4


Run the find_best_configuration first. 
This command will actually give you the inference_instructions.txt file including 
the predict and apply_postprocessing commands. 

2) Run predictions on the test set 
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities

Copy these command from the inference_instructions.txt file 
(don’t type them yourself or you will forget some important arguments!) 
and specify the correct input and output folders. 
Once you have run these two commands you will get your final predictions of the test set. 

Run inference like this 

nnUNetv2_predict -d Dataset050_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/imagesTs -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/Predicted -f  1 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans


3) Postprocessing 

nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE 

***Once inference is completed, run postprocessing like this:***

nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/Predicted -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/Postprocessed -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_1/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_1/plans.json


4) Evaluate predictions 
Run the function evaluate_folder_entry_point after chaning the paths to my predicted test images 

Then you can use the nnUNetv2_evaluate_folder command to evaluate these predictions. 
Type “nnUNetv2_evaluate_folder -h” from the command line to see what input arguments are required. 

nnUNetv2_evaluate_folder -h

nnUNetv2_evaluate_folder -h
usage: nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/dataset.json 
-pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill
/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/labelsTs /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/Predicted


nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset050_IA/labelsTs /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset050_IA/Predicted

positional arguments:
  gt_folder       folder with gt segmentations
  pred_folder     folder with predicted segmentations

options:
  -h, --help      show this help message and exit
  -djfile DJFILE  dataset.json file
  -pfile PFILE    plans.json file
  -o O            Output file. Optional. Default:
                  pred_folder/summary.json
  -np NP          number of processes used. Optional. Default:
                  8
  --chill         dont crash if folder_pred does not have all
                  files that are present in folder_gt

 """
# Predictions 
""" 
Tutorial predictions 
CUDA_VISIBLE_DEVICES=0 nnUNet_predict [...] --part_id 0 --num_parts 2
CUDA_VISIBLE_DEVICES=1 nnUNet_predict [...] --part_id 1 --num_parts 2 """

############################################################################################################

# Steps for the experiment 1 : Predict and evaluate using each of the 5 models 

# fold 0 
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 0
set CUDA_VISIBLE_DEVICES=0 & nnUNetv2_predict command from inference_instructions.txt
***Run inference like this:***

set CUDA_VISIBLE_DEVICES=0 & nnUNetv2_predict -d Dataset057_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/Predicted_internal -f  0 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans

***Once inference is completed, run postprocessing like this:***

nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/Predicted_internal -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/Postprocessed_internal -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0/plans.json

nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/Postprocessed_internal


#fold 1
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 1
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset057_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/predicted_internal_f1 -f 1 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/predicted_internal_f1 -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_f1 -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_1/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_1/plans.json
nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold1/predicted_internal


#fold 2
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 2
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset057_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold2/predicted_internal -f  2 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold2/predicted_internal -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold2/postprocessed_internal -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_2/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_2/plans.json
nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold2/postprocessed_internal


#fold 3
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 3
set CUDA_VISIBLE_DEVICES=0 & nnUNetv2_predict command from inference_instructions.txt
nnUNetv2_apply_postprocessing command from inference_instructions.txt

#fold 4 
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 4
set CUDA_VISIBLE_DEVICES=0 & nnUNetv2_predict command from inference_instructions.txt
nnUNetv2_apply_postprocessing command from inference_instructions.txt


# Steps for the experiment 2 : Predict and evaluate using the best of the 5 models (according to nnunet) 

nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 0 1 2 3 4
set CUDA_VISIBLE_DEVICES=0 & nnUNetv2_predict command from inference_instructions.txt
nnUNetv2_apply_postprocessing command from inference_instructions.txt


# Steps for the experiment 3 : Train for 2000 epochs, the first fold only to see if it improves the results

export PYTHONPATH="/home/golubeka/Master_thesis/nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length"

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 057 3d_fullres 0 -tr nnUNetTrainer_2000epochs


############################ Automate execution of commands 


# Write the commands,  make sure they execute one after the other only if there's no error in the previous command by setting set -e flag


def generate_script(dataset_id, fold_number, test_dataset):
    commands = [
        "#!/bin/bash",
        "set -e",
        f'echo "Finding best configuration for fold {fold_number}"',
        f'nnUNetv2_find_best_configuration {dataset_id} -c 3d_fullres -f {fold_number}',
        f'echo "Predicting {test_dataset} test set fold {fold_number}"',
        f"CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset{dataset_id}_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset{dataset_id}_IA/imagesTs_{test_dataset} -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/predicted_{test_dataset}_f{fold_number} -f  {fold_number} -tr nnUNetTrainer_2000epochs -c 3d_fullres -p nnUNetPlans",
        f'echo "Predicting {test_dataset} completed successfully"',
        f'echo "Postprocessing {test_dataset} test set fold {fold_number}"',
        f"nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/predicted_{test_dataset}_f{fold_number} -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/postprocessed_{test_dataset}_f{fold_number} -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/nnUNetTrainer_2000epochs_nnUNetPlans__3d_fullres/crossval_results_folds_{fold_number}/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/nnUNetTrainer_2000epochs_nnUNetPlans__3d_fullres/crossval_results_folds_{fold_number}/plans.json",
        f'echo "Postprocessing {test_dataset} completed successfully"',
        f'echo "Evaluation {test_dataset} test set fold {fold_number}"',
        f"nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset{dataset_id}_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/nnUNetTrainer_2000epochs_nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset{dataset_id}_IA/labelsTs_{test_dataset} /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/postprocessed_{test_dataset}_f{fold_number}",
        f'echo "Evaluation {test_dataset} completed successfully"'
    ]

    script_filename = f'script_{dataset_id}_{fold_number}_{test_dataset}.sh'
    with open(script_filename, 'w') as f:
        for command in commands:
            f.write(command + '\n')

# Use the function
#generate_script('057', '1', 'internal')
#generate_script('057', '2', 'internal')
generate_script('057', '3', 'internal')
generate_script('057', '4', 'internal')
generate_script('057', '0', 'internal')

bash predict_evaluate_all.sh # Run the general script
# it will run all the prediction-postprocessing-evaluation scripts one after the other for all folds 


