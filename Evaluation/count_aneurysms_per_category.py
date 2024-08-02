""" After defining the relevant size categories <5, 5-10, >10, 
I want to count how many aneurysms there are in each category for each test set ( MR and CT)"""


from matching_v3 import compute_distances, compute_metrics, label_components, load_nifti, match_components
from matching_v3_filter1mm import compute_properties  # filtering <1mm aneurysms
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib

def categorize_aneurysm(size):
    if size < 5:
        return '<5mm'
    elif 5 <= size <= 10:
        return '5-10mm'
    else:
        return '>10mm'

def count_aneurysms_per_size(gt_files, aggregated_csv_file_name):
    """ Process multiple NIfTI file pairs to calculate and aggregate metrics. """
    size_categories = ['<5mm', '5-10mm', '>10mm']
    gt_counts = {cat: 0 for cat in size_categories}
    
    for gt_file in tqdm(gt_files, total=len(gt_files), desc="Processing Files"):
        gt_data, gt_voxel_size = load_nifti(gt_file)
        gt_labeled, gt_num_labels = label_components(gt_data)
        gt_props = compute_properties(gt_labeled, gt_num_labels, gt_voxel_size)

        # Update ground truth counts
        for gt in gt_props:
            size = 2 * gt['radius_mm']
            category = categorize_aneurysm(size)
            gt_counts[category] += 1

    aggregated_metrics = []
    for category in size_categories:
        aggregated_metrics.append({
            "Size Category": category,
            "GT Count": gt_counts[category]
        })

    df_aggregated = pd.DataFrame(aggregated_metrics)
    df_aggregated.to_csv(aggregated_csv_file_name, index=False)
    print(f"Aggregated results saved to {aggregated_csv_file_name}")
    return df_aggregated

if __name__ == "__main__":
    """ dataset = "MR"

    # Process test files
    test_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
    test_files = [os.path.join(test_files_path, f) for f in os.listdir(test_files_path) if f.endswith('.nii.gz')]
    aggregated_csv_file_name_test = f"{dataset}_count_an_per_size_test.csv"
    count_aneurysms_per_size(test_files, aggregated_csv_file_name_test)
    print(f"Processed {dataset} test ")

    # Load the aggregated data for training and internal test sets
    test_df = pd.read_csv(aggregated_csv_file_name_test)

    # Set the size category as a categorical type with a specified order
    size_categories = ['<5mm', '5-10mm', '>10mm']
    test_df['Size Category'] = pd.Categorical(test_df['Size Category'], categories=size_categories, ordered=True)


    # Aggregate data by size category
    test_counts = test_df.groupby('Size Category')['GT Count'].sum().reset_index()

    # Normalize counts to get percentages
    test_total = test_counts['GT Count'].sum()
    test_counts['Percentage'] = (test_counts['GT Count'] / test_total) * 100
 """

    dataset = "CT"
    # Process test files
    test_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'
    test_files = [os.path.join(test_files_path, f) for f in os.listdir(test_files_path) if f.endswith('.nii.gz')]
    aggregated_csv_file_name_test = f"{dataset}_count_an_per_size_test.csv"
    count_aneurysms_per_size(test_files, aggregated_csv_file_name_test)
    print(f"Processed {dataset} test ")

    # Plot for CT
    # Load the aggregated data for training and internal test sets
    test_df = pd.read_csv(aggregated_csv_file_name_test)

    # Set the size category as a categorical type with a specified order
    size_categories = ['<5mm', '5-10mm', '>10mm']
    test_df['Size Category'] = pd.Categorical(test_df['Size Category'], categories=size_categories, ordered=True)


    # Aggregate data by size category
    test_counts = test_df.groupby('Size Category')['GT Count'].sum().reset_index()

    # Normalize counts to get percentages
    test_total = test_counts['GT Count'].sum()
    test_counts['Percentage'] = (test_counts['GT Count'] / test_total) * 100

    