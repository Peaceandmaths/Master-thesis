#!/bin/bash

set -e

# Evaluation
echo "Evaluation internal test all folds "
nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_all_folds
echo "Evaluation completed successfully"

