#!/bin/bash

# Directory paths
input_images_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTr"
output_images_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_brain/imagesTr"

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

   # Apply TotalSegmentator on image
  TotalSegmentator -i "$image_path" -o "$output_image_path" -ta total --roi_subset brain --device gpu --statistics
  # Check if TotalSegmentator succeeded
    if [ $? -ne 0 ]; then
        echo "TotalSegmentator failed for $image_path"
        exit 1
    else
        echo "TotalSegmentator finished successfully for $image_path"
    fi

done