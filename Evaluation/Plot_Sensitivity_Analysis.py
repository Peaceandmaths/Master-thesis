from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import os
from tqdm import tqdm

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0
    return intersection / union

def compute_tpr_fpr_at_iou_threshold(gt_files, pred_files, iou_threshold):
    """
    Compute the true positive rate (TPR) and false positive rate (FPR) based on connected components for a list of prediction and ground truth NIfTI files.

    Parameters:
    - gt_files: list of ground truth NIfTI file paths.
    - pred_files: list of prediction NIfTI file paths.
    - iou_threshold: the IoU threshold for considering a detection as true positive.

    Returns:
    - tpr: the true positive rate at the specified IoU threshold.
    - fpr: the false positive rate at the specified IoU threshold.
    """
    tp_count = 0
    fn_count = 0
    fp_count = 0
    total_pred_count = 0

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total = len(gt_files), desc = "Computing tpr and fpr"):
        pred_img = nib.load(pred_file)
        gt_img = nib.load(gt_file)

        pred_data = (pred_img.get_fdata() > 0.5).astype(int)
        gt_data = (gt_img.get_fdata() > 0.5).astype(int)

        pred_labels, pred_n_labels = ndimage.label(pred_data)
        gt_labels, gt_n_labels = ndimage.label(gt_data)

        used_gt_labels = set()
        total_pred_count += pred_n_labels

        for pred_label in range(1, pred_n_labels + 1):
            pred_comp = pred_labels == pred_label
            ious = [compute_iou(pred_comp, gt_labels == gt_label) if gt_label not in used_gt_labels else 0 for gt_label in range(1, gt_n_labels + 1)]
            
            if ious:
                max_iou = max(ious)
                max_index = ious.index(max_iou)
                gt_match = max_index + 1 if max_iou >= iou_threshold else None
            else:
                gt_match = None

            if gt_match:
                used_gt_labels.add(gt_match)
                tp_count += 1
            else:
                fp_count += 1

        matched_gt = {gt_label for gt_label in used_gt_labels}
        fn_count += gt_n_labels - len(matched_gt)

    tpr = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    fpr = fp_count / total_pred_count if total_pred_count > 0 else 0

    return tpr, fpr


def compute_roc_analysis(gt_files, pred_files):

    iou_values = np.arange(0.1, 1.1, 0.1)
    tpr_values = []  # True Positive Rate
    fpr_values = []  # False Positive Rate

    for iou in tqdm(iou_values, desc = "Computing ious", total = len(iou_values)):
        # Here you need to compute the true positive rate (tpr) and false positive rate (fpr)
        # based on your ground truth (gt_files) and predicted files (pred_files) at the current IoU threshold.
        # This part highly depends on your `compute_validation` function and how you calculate TPR and FPR.
        tpr,fpr = compute_tpr_fpr_at_iou_threshold(gt_files, pred_files, iou)
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    # Plotting
    plt.figure()
    plt.plot(iou_values, tpr_values, label='True Positive Rate')
    plt.plot(iou_values, fpr_values, label='False Positive Rate')
    plt.xlabel('IoU')
    plt.ylabel('Rate')
    plt.title('ROC Analysis')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the figure before showing it
    plt.savefig('roc_analysis_CT_f2.png', dpi=300)  # Save as PNG
    print("ROC analysis saved to roc_analysis_MR_f2.png")

    plt.show()

    # Find intersection
    intersect_idx = np.argwhere(np.diff(np.sign(np.array(fpr_values) - np.array(tpr_values)))).flatten()
    print(f"Intersection is at IoU = {iou_values[intersect_idx]}")


if __name__ == "__main__":
    pred_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f2'
    gt_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
    gt_files = [os.path.join(gt_files_path, f) for f in os.listdir(gt_files_path) if f.endswith('.nii.gz')]
    pred_files = [os.path.join(pred_files_path, f) for f in os.listdir(pred_files_path) if f.endswith('.nii.gz')]
    
    compute_roc_analysis(gt_files, pred_files)
