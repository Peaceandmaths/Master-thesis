""" Matching_v3 but I filter out anuerysms smaller than 1mm 
Target wise metrics using center of mass criterion  """ 

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from medpy.metric import hd95
from tqdm import tqdm

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.header.get_zooms()

def label_components(data):
    "Counts and extracts aneurysms"
    labeled_data, num_labels = ndimage.label(data)
    return labeled_data, num_labels

def compute_properties(labeled_data, num_labels, voxel_size):
    properties = []
    for i in range(1, num_labels + 1):
        component = (labeled_data == i)
        coords = np.transpose(np.nonzero(component))
        if coords.size == 0:
            continue
        center_of_mass = coords.mean(axis=0)
        radius_voxels = np.linalg.norm(coords - center_of_mass, axis=1).max()
        radius_mm = radius_voxels * voxel_size[0]  # Assuming isotropic voxels
        if radius_mm < 0.5: #filter out aneurysms smaller than 1mm
            continue
        properties.append({'id': i, 'center_of_mass': center_of_mass, 'radius_voxels': radius_voxels, 'radius_mm': radius_mm, 'component': component})
    return properties

def compute_distances(pred_props, gt_props):
    distances = np.zeros((len(pred_props), len(gt_props)))
    for i, pred in enumerate(pred_props):
        for j, gt in enumerate(gt_props):
            distances[i, j] = np.linalg.norm(pred['center_of_mass'] - gt['center_of_mass'])
    return distances

def match_components(pred_props, gt_props, distances):
    matched = []
    for i, pred in enumerate(pred_props):
        for j, gt in enumerate(gt_props):
            if distances[i, j] < (pred['radius_voxels'] + gt['radius_voxels']):
                matched.append((i, j))
    return matched

def compute_metrics(pred_component, gt_component):
    iou = np.sum(pred_component & gt_component) / np.sum(pred_component | gt_component)
    hausdorff = hd95(pred_component, gt_component)
    dsc = 2 * np.sum(pred_component & gt_component) / (np.sum(pred_component) + np.sum(gt_component))
    return iou, hausdorff, dsc

def process_files(pred_files, gt_files):
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
                results.append({'file_name': os.path.basename(gt_file), 'pred_id': None, 'gt_id': gt['id'], 'IoU': 0, 'Hausdorff': 'N/A', 'DSC': 0, 'Match Type': 'FN'})
            continue

        if not gt_props:
            for pred in pred_props:
                results.append({'file_name': os.path.basename(pred_file), 'pred_id': pred['id'], 'gt_id': None, 'IoU': 0, 'Hausdorff': 'N/A', 'DSC': 0, 'Match Type': 'FP'})
            continue

        distances = compute_distances(pred_props, gt_props)
        matches = match_components(pred_props, gt_props, distances)

        matched_pred_ids = set()
        matched_gt_ids = set()

        for i, j in matches:
            iou, hausdorff, dsc = compute_metrics(pred_props[i]['component'], gt_props[j]['component'])
            results.append({'file_name': os.path.basename(pred_file), 'pred_id': pred_props[i]['id'], 'gt_id': gt_props[j]['id'], 'IoU': iou, 'Hausdorff': hausdorff, 'DSC': dsc, 'Match Type': 'TP'})
            matched_pred_ids.add(pred_props[i]['id'])
            matched_gt_ids.add(gt_props[j]['id'])

        for pred in pred_props:
            if pred['id'] not in matched_pred_ids:
                results.append({'file_name': os.path.basename(pred_file), 'pred_id': pred['id'], 'gt_id': None, 'IoU': 0, 'Hausdorff': 'N/A', 'DSC': 0, 'Match Type': 'FP'})

        for gt in gt_props:
            if gt['id'] not in matched_gt_ids:
                results.append({'file_name': os.path.basename(gt_file), 'pred_id': None, 'gt_id': gt['id'], 'IoU': 0, 'Hausdorff': 'N/A', 'DSC': 0, 'Match Type': 'FN'})

    results_df = pd.DataFrame(results)
    results_df.to_csv("detection_metrics.csv", index=False)
    return results_df

def aggregate_metrics(results_df):
    tp = len(results_df[results_df['Match Type'] == 'TP'])
    fp = len(results_df[results_df['Match Type'] == 'FP'])
    fn = len(results_df[results_df['Match Type'] == 'FN'])
    tn = len(results_df[results_df['Match Type'] == 'TN']) #number of cases with no aneurysm in both gt and pred

    total_cases = results_df['file_name'].nunique()
    mean_tp = tp / total_cases
    mean_fp = fp / total_cases
    mean_fn = fn / total_cases
    #mean_tn = tn / total_cases

    mean_iou = results_df[results_df['Match Type'] == 'TP']['IoU'].mean()
    mean_hausdorff = results_df[results_df['Match Type'] == 'TP']['Hausdorff'].replace('N/A', np.nan).astype(float).mean()
    mean_dsc = results_df[results_df['Match Type'] == 'TP']['DSC'].mean()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0 # also computes dice when there's no match (0) 

    aggregated_metrics = {
        "TPs": tp,
        "FPs": fp,
        "FNs": fn,
        "Mean TP per case": mean_tp,
        "Mean FP per case": mean_fp,
        "Mean FN per case": mean_fn,
        "Mean IoU": mean_iou,
        "Mean Hausdorff": mean_hausdorff,
        "Mean DSC": mean_dsc,
        "Precision": precision,
        "Recall/Sensitivity": recall,
        "DSC": dsc
        } # also computes dice when there's no match (0) 
    

    return aggregated_metrics

if __name__ == "__main__":
    pred_files_path = '/path/to/predicted/files'
    gt_files_path = '/path/to/ground_truth/files'
    gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
    pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
    
    results_df = process_files(pred_files, gt_files)
    aggregated_metrics = aggregate_metrics(results_df)
    aggregated_metrics_df = pd.DataFrame([aggregated_metrics])
    aggregated_metrics_df.to_csv("aggregated_metrics.csv", index=False)
    print("Aggregated results saved to aggregated_metrics.csv")



