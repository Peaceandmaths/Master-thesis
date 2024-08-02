#!/bin/bash

# Exit on error
set -e

# Set environment variables
export nnUNet_raw="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"  
export nnUNet_results="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"
export nnUNet_preprocessed="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"

# Training the model
for i in {0..4}
do
    echo "Starting training for fold $i..."
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 059 3d_fullres $i --npz
    echo "Training for fold $i done."
done

echo "Training is done."

# Finding best configuration
echo "Finding best configuration..."
nnUNetv2_find_best_configuration 059 -c 3d_fullres -f 0 1 2 3 4
echo "Best configuration found. See instructions file."