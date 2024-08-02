from Trying_AUC_ROC import compute_validation
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def process_files_with_paths(pred_files, gt_files, aggregated_csv_file_name, component_wise_csv_file_name):
    """ Process multiple NIfTI file pairs to calculate and aggregate metrics. """
    all_results = []
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        file_results = compute_validation(pred_file, gt_file)
        all_results.extend(file_results)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(component_wise_csv_file_name, index=False)
    print(f"Component-wise results saved to {component_wise_csv_file_name}")

    # Aggregate metrics
    tp = df[df['Match Type'] == 'TP'].shape[0]
    fp = df[df['Match Type'] == 'FP'].shape[0]
    fn = df[df['Match Type'] == 'FN'].shape[0]

    total_cases = len(pred_files)  # Assuming each file represents a case
    mean_tp = tp / total_cases
    mean_fp = fp / total_cases
    mean_fn = fn / total_cases

    mean_iou = df[df['Match Type'] == 'TP']['IoU'].mean()
    mean_hausdorff = df[df['Match Type'] == 'TP']['Hausdorff'].replace('N/A', np.nan).astype(float).mean()
    
    auc_values = [result['AUC'] for result in all_results if result['AUC'] is not None]
    mean_auc = sum(auc_values) / len(auc_values) if auc_values else None

    ap_values = [result['AP'] for result in all_results if result['AP'] is not None]
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else None

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall  # Same as recall
    dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    

    # Save aggregated metrics
    aggregated_metrics = {
        "Mean TP per case": mean_tp,
        "Mean FP per case": mean_fp,
        "Mean FN per case": mean_fn,
        "Mean IoU": mean_iou,
        "Mean Hausdorff": mean_hausdorff,
        "Precision": precision,
        "Recall": recall,
        "Sensitivity": sensitivity,
        "DSC": dsc,
        "Mean AUC": mean_auc,  # Add this line
        "Mean AP": mean_ap  # Add this line
        
    }
    df_aggregated = pd.DataFrame([aggregated_metrics])
    df_aggregated.to_csv(aggregated_csv_file_name, index=False)
    print(f"Aggregated results saved to {aggregated_csv_file_name}")


if __name__ == "__main__":
    dataset = "MR"
    folds = range(5)
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}.csv"
            component_wise_csv_file_name = f"component_wise_{dataset}_f{i}.csv"
            process_files_with_paths(pred_files, gt_files, aggregated_csv_file_name, component_wise_csv_file_name)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
        
    dataset = "CT"
    folds = range(5)
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset059_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}.csv"
            component_wise_csv_file_name = f"component_wise_{dataset}_f{i}.csv"
            process_files_with_paths(pred_files, gt_files, aggregated_csv_file_name, component_wise_csv_file_name)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
