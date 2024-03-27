#!/bin/bash
set -e
echo "Finding best configuration for fold 3"
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 3
echo "Postprocessing internal test set fold 3"
nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/predicted_internal_f3 -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_f3 -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_3/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_3/plans.json
echo "Postprocessing internal completed successfully"
echo "Evaluation internal test set fold 3"
nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_f3
echo "Evaluation internal completed successfully"
