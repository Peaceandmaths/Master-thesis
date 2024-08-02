#!/bin/bash

# The list of Python scripts to run for evaluation
# All these scripts run on MR and CT, over all 5 folds 
# and create tables with means and 95% confidence intervals 

scripts=(
    "compute_all_iou_after_filtering.py" # target-wise segmentation based on iou thresholding method 
    "compute_all_center_of_mass.py" # target-wise segmentation based on center of mass method
    "matching_voxelwise_v2.py" # voxel-wise segmentation 
    "detection_per_size" # detection per size
)

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python "$script"
    if [ $? -ne 0 ]; then
        echo "Error running $script"
        exit 1
    fi
done

echo "All evaluations completed successfully."