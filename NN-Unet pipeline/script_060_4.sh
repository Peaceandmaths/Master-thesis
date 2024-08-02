#!/bin/bash
set -e
echo "Finding best configuration for fold 4"
nnUNetv2_find_best_configuration 060 -c 3d_fullres -f 4
echo "Predicting test set fold 4"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset060_IA -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/imagesTs -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/predicted/predicted_f4 -f  4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
echo "Predicting completed successfully"
echo "Postprocessing test set fold 4"
nnUNetv2_apply_postprocessing -i /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/predicted/predicted_f4 -o /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f4 -pp_pkl_file /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/postprocessing.pkl -np 8 -plans_json /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/plans.json
echo "Postprocessing completed successfully"
echo "Evaluation test set fold 4"
nnUNetv2_evaluate_folder -djfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/dataset.json -pfile /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f4
echo "Evaluation MR data 4 completed successfully"
