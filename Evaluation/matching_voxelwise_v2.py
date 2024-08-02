""" In this version of matching_voxelwise I changed the counting condition from ==1 to >0 because apparently data is not boolean. 
Use binarizing code and fix the bug with np.bool that changed in the newest version of numpy 
Summarize table computes 95% Confidence intervals """

import os
import numpy as np
np.bool = np.bool_
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from medpy.metric import hd95

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.header.get_zooms()

""" def safe_hd95(pred_data, gt_data):
    # Convert to bool_ instead of bool
    pred_data = pred_data.astype(bool)
    gt_data = gt_data.astype(bool)
    return hd95(pred_data, gt_data) """

from medpy.metric import binary

# Patch the hd95 function in medpy
""" def patched_hd95(result, reference, voxelspacing=None, connectivity=1):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    # The surface distance is only defined when there is at least one foreground voxel in both
    if 0 == np.count_nonzero(result) or 0 == np.count_nonzero(reference):
        raise RuntimeError("The surface distance is only defined when there is at least one foreground voxel in both images.")

    return binary.hd(result, reference, voxelspacing, connectivity) """

# Override the original hd95 function with the patched version
# binary.hd95 = patched_hd95

def compute_metrics(pred_data, gt_data):
    """ tp = np.sum(((pred_data == 1) & (gt_data == 1)))
    fp = np.sum(((pred_data == 1) & (gt_data == 0)))
    fn = np.sum(((pred_data == 0) & (gt_data == 1))) """

    pred_data = (pred_data > 0.5).astype(int)  # Assume threshold to binary
    gt_data = (gt_data > 0.5).astype(int) 
    
    tp = np.sum((pred_data > 0) & (gt_data > 0))
    fp = np.sum((pred_data > 0) & (gt_data == 0))
    fn = np.sum((pred_data == 0) & (gt_data > 0))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    hausdorff = hd95(pred_data, gt_data) if tp > 0 else np.nan
    dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return tp, fp, fn, iou, hausdorff, dsc

def process_files(pred_files, gt_files, vowel_wise_csv_file_name):
    results = []

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        pred_data, pred_voxel_size = load_nifti(pred_file)
        gt_data, gt_voxel_size = load_nifti(gt_file)

        pred_data = (pred_data > 0.5).astype(int)  # Assume threshold to binary
        gt_data = (gt_data > 0.5).astype(int)    

        # Early treatment of true negative images
        if (np.sum(pred_data) == 0 and np.sum(gt_data) == 0):
            results.append({'file_name': os.path.basename(pred_file), 'Match Type': 'TN'})
            continue

        # Calculate metrics
        tp, fp, fn, iou, hausdorff, dsc = compute_metrics(pred_data, gt_data)

        if tp > 0:
            results.append({'file_name': os.path.basename(pred_file), 'TP': tp, 'FP': fp, 'FN': fn, 'IoU': iou, 'Hausdorff': hausdorff, 'DSC': dsc, 'Match Type': 'TP'})
        else:
            if fp > 0:
                results.append({'file_name': os.path.basename(pred_file), 'TP': tp, 'FP': fp, 'FN': fn, 'IoU': iou, 'Hausdorff': 'N/A', 'DSC': dsc, 'Match Type': 'FP'})
            if fn > 0:
                results.append({'file_name': os.path.basename(pred_file), 'TP': tp, 'FP': fp, 'FN': fn, 'IoU': iou, 'Hausdorff': 'N/A', 'DSC': dsc, 'Match Type': 'FN'})

    results_df = pd.DataFrame(results)
    results_df.to_csv(vowel_wise_csv_file_name, index=False)
    print(f"Voxel-wise results saved to {vowel_wise_csv_file_name}")
    return results_df

def aggregate_metrics(results_df, aggregated_csv_file_name):
    tp = results_df['TP'].sum()
    fp = results_df['FP'].sum()
    fn = results_df['FN'].sum()
    
    total_cases = results_df['file_name'].nunique()
    mean_tp = tp / total_cases
    mean_fp = fp / total_cases
    mean_fn = fn / total_cases

    mean_iou = results_df[results_df['Match Type'] == 'TP']['IoU'].mean()
    mean_hausdorff = results_df[results_df['Match Type'] == 'TP']['Hausdorff'].replace('N/A', np.nan).astype(float).mean()
    mean_dsc = results_df[results_df['Match Type'] == 'TP']['DSC'].mean()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    aggregated_metrics = {
        "Mean TP per case": mean_tp,
        "Mean FP per case": mean_fp,
        "Mean FN per case": mean_fn,
        "Mean IoU": mean_iou,
        "Mean Hausdorff": mean_hausdorff,
        "Mean DSC": mean_dsc,
        "Precision": precision,
        "Recall/Sensitivity": recall,
        "DSC": dsc
    }

    df_aggregated = pd.DataFrame([aggregated_metrics])
    df_aggregated.to_csv(aggregated_csv_file_name, index=False)
    print(f"Aggregated results saved to {aggregated_csv_file_name}")

    return aggregated_metrics

import scipy.stats

def summarize_metrics_CI(all_metrics_df, dataset):
    metrics_summary = all_metrics_df.agg(['mean', 'std', 'count']).transpose()

    metrics_summary['lower_bound'] = metrics_summary.apply(
        lambda row: row['mean'] - (1.96 * (row['std'] / (row['count'] ** 0.5))), axis=1)
    metrics_summary['upper_bound'] = metrics_summary.apply(
        lambda row: row['mean'] + (1.96 * (row['std'] / (row['count'] ** 0.5))), axis=1)

    metrics_summary['formatted_result'] = metrics_summary.apply(
        lambda row: f"{row['mean']:.2f} ({row['lower_bound']:.2f} - {row['upper_bound']:.2f})", axis=1)

    metrics_summary['formatted_result'].to_csv(f"formatted_summary_{dataset}_95CI_voxelwise.csv", header=True)
    print(f"Formatted summary saved to formatted_summary_{dataset}_95CI_voxelwise.csv")

if __name__ == "__main__":
    """ dataset = "MR"
    all_metrics_MR = [] # List to store aggregated metrics for all folds
    folds = range(5)
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}_voxelwise.csv"
            component_wise_csv_file_name = f"voxelwise_{dataset}_f{i}.csv"
            results_df = process_files(pred_files, gt_files, component_wise_csv_file_name)
            aggregated_metrics = aggregate_metrics(results_df, aggregated_csv_file_name)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
            aggregated_metrics_df = pd.DataFrame([aggregated_metrics])  # Convert dict to DataFrame
            all_metrics_MR.append(aggregated_metrics_df)  # Append DataFrame instead of dict

    all_metrics_df = pd.concat(all_metrics_MR, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_metrics_{dataset}_voxelwise_CI.csv", index=False)
    summarize_metrics_CI(all_metrics_df, dataset) """

    dataset = "CT"
    all_metrics_CT = [] # List to store aggregated metrics for all folds
    folds = range(5)
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset059_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            aggregated_csv_file_name = f"aggregated_{dataset}_f{i}_voxelwise.csv"
            component_wise_csv_file_name = f"voxelwise_{dataset}_f{i}.csv"
            results_df = process_files(pred_files, gt_files, component_wise_csv_file_name)
            aggregated_metrics = aggregate_metrics(results_df, aggregated_csv_file_name)
            print(f"Processed {dataset} fold {i}")
            aggregated_metrics_df = pd.DataFrame([aggregated_metrics])  # Convert dict to DataFrame
            all_metrics_CT.append(aggregated_metrics_df)  # Append DataFrame instead of dict
            pbar.update()

    all_metrics_df = pd.concat(all_metrics_CT, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_metrics_{dataset}_voxelwise_CI.csv", index=False)
    summarize_metrics_CI(all_metrics_df, dataset)



