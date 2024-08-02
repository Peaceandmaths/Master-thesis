from matching_v3 import compute_distances, compute_metrics, label_components, load_nifti, match_components
from matching_v3_filter1mm import compute_properties  # filtering <1mm aneurysms
import os
import pandas as pd
import numpy as np
np.bool = np.bool_
from tqdm import tqdm
from detection_per_size_summary import compute_detection_per_size_ci


def categorize_aneurysm(size):
    if size < 5:
        return '<5mm'
    elif 5 <= size <= 10:
        return '5-10mm'
    else:
        return '>10mm'

def process_files(pred_files, gt_files, component_wise_csv_file_name):
    """ Process multiple NIfTI file pairs to calculate and aggregate metrics. """
    results = []

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        pred_data, pred_voxel_size = load_nifti(pred_file)
        gt_data, gt_voxel_size = load_nifti(gt_file)

        pred_labeled, pred_num_labels = label_components(pred_data)
        gt_labeled, gt_num_labels = label_components(gt_data)

        pred_props = compute_properties(pred_labeled, pred_num_labels, pred_voxel_size)
        gt_props = compute_properties(gt_labeled, gt_num_labels, gt_voxel_size)

        # Handle True Negatives
        if not pred_props and not gt_props:
            results.append({'file_name': os.path.basename(pred_file), 'Match Type': 'TN'})
            continue

        # Skip further computation if there are no predictions or no ground truths
        if not pred_props:
            for gt in gt_props:
                size = 2 * gt['radius_mm']
                category = categorize_aneurysm(size)
                results.append({'file_name': os.path.basename(gt_file), 'pred_id': None, 'gt_id': gt['id'], 'Match Type': 'FN', 'Size Category': category})
            continue

        if not gt_props:
            for pred in pred_props:
                size = 2 * pred['radius_mm']
                category = categorize_aneurysm(size)
                results.append({'file_name': os.path.basename(pred_file), 'pred_id': pred['id'], 'gt_id': None, 'Match Type': 'FP', 'Size Category': category})
            continue

        distances = compute_distances(pred_props, gt_props)
        matches = match_components(pred_props, gt_props, distances)

        matched_pred_ids = set()
        matched_gt_ids = set()

        for i, j in matches:
            size = 2 * gt_props[j]['radius_mm']
            category = categorize_aneurysm(size)
            results.append({'file_name': os.path.basename(pred_file), 'pred_id': pred_props[i]['id'], 'gt_id': gt_props[j]['id'], 'Match Type': 'TP', 'Size Category': category})
            matched_pred_ids.add(pred_props[i]['id'])
            matched_gt_ids.add(gt_props[j]['id'])

        for pred in pred_props:
            if pred['id'] not in matched_pred_ids:
                size = 2 * pred['radius_mm']
                category = categorize_aneurysm(size)
                results.append({'file_name': os.path.basename(pred_file), 'pred_id': pred['id'], 'gt_id': None, 'Match Type': 'FP', 'Size Category': category})

        for gt in gt_props:
            if gt['id'] not in matched_gt_ids:
                size = 2 * gt['radius_mm']
                category = categorize_aneurysm(size)
                results.append({'file_name': os.path.basename(gt_file), 'pred_id': None, 'gt_id': gt['id'], 'Match Type': 'FN', 'Size Category': category})

    results_df = pd.DataFrame(results)
    #results_df.to_csv(component_wise_csv_file_name, index=False)
    #print(f"Component-wise results saved to {component_wise_csv_file_name}")
    return results_df

def aggregate_metrics(results_df):
    size_categories = ['<5mm', '5-10mm', '>10mm']
    aggregated_metrics = []

    for category in size_categories:
        tp = len(results_df[(results_df['Match Type'] == 'TP') & (results_df['Size Category'] == category)])
        fp = len(results_df[(results_df['Match Type'] == 'FP') & (results_df['Size Category'] == category)])
        fn = len(results_df[(results_df['Match Type'] == 'FN') & (results_df['Size Category'] == category)])

        total_cases = results_df['file_name'].nunique()
        mean_fp = fp / total_cases
        recall = (tp / (tp + fn) if (tp + fn) > 0 else 0) * 100  # Convert recall to percentage

        aggregated_metrics.append({
            "Size Category": category,
            "Recall(%)": f"{recall:.2f}",
            "FPs per case": mean_fp
        })

    df_aggregated = pd.DataFrame(aggregated_metrics)
    return df_aggregated

def summarize_metrics(all_metrics_df, dataset):
    """
    Calculates the mean and standard deviation for each metric in a DataFrame,
    formats the summary as "mean (mean - sd, mean + sd)", and saves the result to a CSV file.

    Parameters:
    - all_metrics_df: A pandas DataFrame containing the metrics to be summarized.

    Returns:
    - None
    """
    # Calculate mean and standard deviation for each metric
    metrics_summary = all_metrics_df.groupby('Size Category').agg(['mean', 'std'])

    # Format the summary
    formatted_summary = metrics_summary.apply(
        lambda x: f"{x['mean']:.2f} ({x['mean'] - x['std']:.2f} - {x['mean'] + x['std']:.2f})", axis=1)

    formatted_summary.to_csv(f"formatted_summary_{dataset}_detection_per_size.csv", header=True)
    print(f"Formatted summary saved to formatted_summary_{dataset}_detection_per_size.csv")



def summarize_metrics_CI(all_metrics_df, dataset):
    metrics_summary = all_metrics_df.agg(['mean', 'std', 'count']).transpose()

    metrics_summary['lower_bound'] = metrics_summary.apply(
        lambda row: row['mean'] - (1.96 * (row['std'] / (row['count'] ** 0.5))), axis=1)
    metrics_summary['upper_bound'] = metrics_summary.apply(
        lambda row: row['mean'] + (1.96 * (row['std'] / (row['count'] ** 0.5))), axis=1)

    metrics_summary['formatted_result'] = metrics_summary.apply(
        lambda row: f"{row['mean']:.2f} ({row['lower_bound']:.2f} - {row['upper_bound']:.2f})", axis=1)

    metrics_summary['formatted_result'].to_csv(f"formatted_summary_{dataset}_95CI_detection_per_size.csv", header=True)
    print(f"Formatted summary saved to formatted_summary_{dataset}_95CI_detection_per_size.csv")




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
            component_wise_csv_file_name = f"component_{dataset}_f{i}_detection_per_size.csv"
            results_df = process_files(pred_files, gt_files, component_wise_csv_file_name)
            aggregated_metrics = aggregate_metrics(results_df)
            all_metrics_MR.append(aggregated_metrics)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
    
    all_metrics_df = pd.concat(all_metrics_MR, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_{dataset}_detection_per_size.csv", index=False)
    summarize_metrics_CI(all_metrics_df, dataset)
    #compute_detection_per_size_ci(all_metrics_df, dataset) """
    
    dataset = "CT"
    all_metrics_CT = [] # List to store aggregated metrics for all folds
    folds = range(5)
    with tqdm(total=len(folds), desc=f"Processing {dataset}") as pbar:
        for i in folds:
            pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset059_IA/postprocessed/postprocessed_f{i}'
            gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'
            gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
            pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
            component_wise_csv_file_name = f"component_{dataset}_f{i}_detection_per_size.csv"
            results_df = process_files(pred_files, gt_files, component_wise_csv_file_name)
            aggregated_metrics = aggregate_metrics(results_df)
            all_metrics_CT.append(aggregated_metrics)
            print(f"Processed {dataset} fold {i}")
            pbar.update()
    
    all_metrics_df = pd.concat(all_metrics_CT, ignore_index=True)
    all_metrics_df.to_csv(f"all_aggregated_{dataset}_detection_per_size.csv", index=False)
    #summarize_metrics_CI(all_metrics_df, dataset)
    #compute_detection_per_size_ci(all_metrics_df, dataset)
    