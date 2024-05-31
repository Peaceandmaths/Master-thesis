
""" Final code to compute all the metrics on CT/MR data, save intermediate results for connected components, 
aggregated mean metrics. It has AUC, AP, Hausdorff distance and all the basic metrics 
(accuracy, precision, recall, sensitivity, DSC). No confusion matrix.

"""

from sklearn.metrics import roc_auc_score, average_precision_score
import os
import numpy as np
import pandas as pd
from scipy import ndimage
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from medpy.metric import hd95
from sklearn.metrics import roc_curve

def compute_metrics(pred, gt):
    """ Returns TP, TN, FP, FN counts for two binary masks. """
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, tn, fp, fn

def compute_iou(pred_comp, gt_comp):
    """ Computes Intersection over Union (IoU) for two binary masks. """
    intersection = np.sum(pred_comp & gt_comp)
    union = np.sum(pred_comp | gt_comp)
    return intersection / union if union != 0 else 0

def compute_hausdorff(pred_comp, gt_comp):
    """ Computes the Hausdorff distance for two binary masks. """
    return hd95(pred_comp, gt_comp)

def compute_validation(pred_mask_file, gt_mask_file):
    """ Compute validation metrics based on connected components for a pair of prediction and ground truth NIfTI files. """
    pred_img = nib.load(pred_mask_file)
    gt_img = nib.load(gt_mask_file)
    
    pred_data = (pred_img.get_fdata() > 0.5).astype(int)  # Assume threshold to binary
    gt_data = (gt_img.get_fdata() > 0.5).astype(int)

    # Check if gt_data contains more than one unique value
    if len(np.unique(gt_data.flatten())) > 1:
        auc = roc_auc_score(gt_data.flatten(), pred_data.flatten())
        ap = average_precision_score(gt_data.flatten(), pred_data.flatten())
    else:
        base_name = os.path.basename(gt_mask_file)
        print(f"Skipping AUC calculation for {base_name} because it contains only one class")
        auc = None
        ap = None

    
    # Label connected components
    pred_labels, pred_n_labels = ndimage.label(pred_data)
    gt_labels, gt_n_labels = ndimage.label(gt_data)

    results = []
    used_gt_labels = set()

    # Extract the case ID from the file name
    base_name = os.path.basename(pred_mask_file)
    case_id = base_name.split('_')[-1].split('.')[0]

    # Process each predicted component
    for pred_label in range(1, pred_n_labels + 1):
        pred_comp = pred_labels == pred_label
        ious = [compute_iou(pred_comp, gt_labels == gt_label) if gt_label not in used_gt_labels else 0 for gt_label in range(1, gt_n_labels + 1)]
        
        if ious:  # Check if any IoU was computed
            max_iou = max(ious)
            max_index = ious.index(max_iou)
            gt_match = max_index + 1 if max_iou >= 0.3 else None  # Matching threshold of IoU >= 0.3
        else:
            max_iou = 0
            gt_match = None

        # Determine if TP, FP, FN
        if gt_match:
            gt_comp = gt_labels == gt_match
            used_gt_labels.add(gt_match)
            hausdorff_dist = compute_hausdorff(pred_comp, gt_comp)
            results.append({'Case ID': case_id, 'Pred Component': pred_label, 'GT Component': gt_match, 'IoU': max_iou, 'Hausdorff': hausdorff_dist,'AUC': auc, 'AP': ap, 'Match Type': 'TP'})
        else:
            results.append({'Case ID': case_id, 'Pred Component': pred_label, 'GT Component': 'None', 'IoU': max_iou, 'Hausdorff': 'N/A','AUC': auc, 'AP': ap, 'Match Type': 'FP'})

    # Check for any missed ground truth components (FN)
    matched_gt = {res['GT Component'] for res in results if res['GT Component'] != 'None'}
    for gt_label in range(1, gt_n_labels + 1):
        if str(gt_label) not in matched_gt:
            results.append({'Case ID': case_id, 'Pred Component': 'None', 'GT Component': gt_label, 'IoU': 0, 'Hausdorff': 'N/A','AUC': auc, 'AP': ap, 'Match Type': 'FN'})

    return results

def process_files(pred_files, gt_files):
    """ Process multiple NIfTI file pairs to calculate and aggregate metrics. """
    all_results = []
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Processing Files"):
        file_results = compute_validation(pred_file, gt_file)
        all_results.extend(file_results)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv("component_wise_MR_f0.csv", index=False)
    print("Component-wise results saved to component_wise_MR_f0.csv")

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
    df_aggregated.to_csv("aggregated_MR_f0.csv", index=False)
    print("Aggregated results saved to aggregated_MR_f0.csv")


if __name__ == "__main__":
        
    pred_files_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f0'
    gt_files_path = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
    gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
    pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
    process_files(pred_files, gt_files)













