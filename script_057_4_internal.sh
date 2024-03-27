#!/bin/bash
set -e
echo "Finding best configuration for fold 4"
nnUNetv2_find_best_configuration 057 -c 3d_fullres -f 4
echo "Predicting internal test set fold 4"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset057_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/predicted_internal_f4 -f  4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
echo "Predicting internal completed successfully"
echo "Postprocessing internal test set fold 4"
nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/predicted_internal_f4 -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_f4 -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/plans.json
echo "Postprocessing internal completed successfully"
echo "Evaluation internal test set fold 4"
nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA/postprocessed_internal_f4
echo "Evaluation internal completed successfully"
