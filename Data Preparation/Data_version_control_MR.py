# Download MR data from HPC shared directory 

# Initiate the file transfert from your GPU machine:
ssh golubeka@160.85.79.231
rsync -aviP golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu-lausanne/derivatives/manual_masks /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels


# Removing non nii.gz files 

import os

def clean_directory_of_non_nii_gz_files(root_dir):
    for dirpath, dirnames, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.nii.gz'):
                full_path = os.path.join(dirpath, file)
                os.remove(full_path)
                print(f"Removed: {full_path}")

# Replace 'your_root_directory_path' with the path to your dataset root directory
root_directory = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/manual_masks'
clean_directory_of_non_nii_gz_files(root_directory)

print("Cleanup complete. All non-.nii.gz files have been removed.")

# Moving nii.gz files to the folder Images_and_labels 

find /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/manual_masks -type f -name "*.nii.gz" -exec mv {} /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels\;


# Control participants and healthy controls ids

Checking_extra_participants.py

# Move control participants to Dataset060_Mr 

bash Move_control_participants.sh

# Create empty labels for control participants

from Creating_empty_labels_MR import create_and_check_empty_mask 
src_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr'
dest_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr'
# Call the function
create_and_check_empty_mask(src_dir, dest_dir)


# Move patients from Images_and_labels to Dataset060_MR after merging lesions 

Merge_lesions.py 
extract_patient_ids.py 
bash Move_patients.sh 

# Remove duplicates 

from Checking_extra_participants import duplicate_ids
bash Remove_duplicates.sh 

## Preprocess and training 


## Find best configuration, Predict and evaluate using 5 models 

def generate_script(dataset_id, fold_number):
    commands = [
        "#!/bin/bash",
        "set -e",
        f'echo "Finding best configuration for fold {fold_number}"',
        f'nnUNetv2_find_best_configuration {dataset_id} -c 3d_fullres -f {fold_number}',
        f'echo "Predicting test set fold {fold_number}"',
        f"CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset{dataset_id}_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset{dataset_id}_IA/imagesTs -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/predicted/predicted_f{fold_number} -f  {fold_number} -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans",
        f'echo "Predicting completed successfully"',
        f'echo "Postprocessing test set fold {fold_number}"',
        f"nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/predicted/predicted_f{fold_number} -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/postprocessed/postprocessed_f{fold_number} -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_{fold_number}/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_{fold_number}/plans.json",
        f'echo "Postprocessing completed successfully"',
        f'echo "Evaluation test set fold {fold_number}"',
        f"nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset{dataset_id}_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset{dataset_id}_IA/labelsTs /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset{dataset_id}_IA/postprocessed/postprocessed_f{fold_number}",
        f'echo "Evaluation MR data {fold_number} completed successfully"'
    ]

    script_filename = f'script_{dataset_id}_{fold_number}.sh'
    with open(script_filename, 'w') as f:
        for command in commands:
            f.write(command + '\n')


# Use the function
generate_script('060', '0')
generate_script('060', '1')
generate_script('060', '2')
generate_script('060', '3')
generate_script('060', '4')

bash predict_evaluate_all.sh
