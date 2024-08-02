#!/bin/bash

# This script moves the original image files of patients  from the source directory to the destination directory.
#!/bin/bash

src_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels"
dest_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr"
source_path="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr"

# Call the Python script to get the patient IDs
patient_ids=$(python3 extract_patient_ids.py $source_path)

# Move the files for each patient ID
for id in $patient_ids; do
   mv "$src_dir/${id}_ses-"*"_desc-brain_mask.nii.gz" "$dest_dir/"
done