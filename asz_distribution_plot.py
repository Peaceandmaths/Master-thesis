from matching_v3 import compute_distances, compute_metrics, label_components, load_nifti, match_components
from matching_v3_filter1mm import compute_properties  # filtering <1mm aneurysms
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib

def categorize_aneurysm(size):
    if size < 3:
        return '<3mm'
    elif 3 < size <= 4:
        return '3-4mm'
    elif 4 < size <= 5:
        return '4-5mm'
    elif 5 < size <= 6:
        return '5-6mm'
    elif 6 < size <= 7:
        return '6-7mm'
    elif 7 < size <= 8:
        return '7-8mm'
    elif 8 < size <= 9:
        return '8-9mm'
    elif 9 < size <= 10:
        return '9-10mm'
    elif 10 < size <= 11:
        return '10-11mm'
    elif 11 < size <= 12:
        return '11-12mm'
    else:
        return '>12mm'

def count_aneurysms_per_size(gt_files, aggregated_csv_file_name):
    """ Process multiple NIfTI file pairs to calculate and aggregate metrics. """
    size_categories = ['<3mm', '3-4mm', '4-5mm', '5-6mm', '6-7mm', '7-8mm', '8-9mm', '9-10mm', '10-11mm', '11-12mm', '>12mm']
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
    dataset = "MR"

    # Process training files
    train_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTr'
    train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path) if f.endswith('.nii.gz')]
    aggregated_csv_file_name_train = f"{dataset}_count_an_per_size_train.csv"
    count_aneurysms_per_size(train_files, aggregated_csv_file_name_train)
    print(f"Processed {dataset} train ")

    # Process test files
    test_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs'
    test_files = [os.path.join(test_files_path, f) for f in os.listdir(test_files_path) if f.endswith('.nii.gz')]
    aggregated_csv_file_name_test = f"{dataset}_count_an_per_size_test.csv"
    count_aneurysms_per_size(test_files, aggregated_csv_file_name_test)
    print(f"Processed {dataset} test ")

    # Plot for MR
    # Load the aggregated data for training and internal test sets
    train_df = pd.read_csv(aggregated_csv_file_name_train)
    test_df = pd.read_csv(aggregated_csv_file_name_test)

    # Set the size category as a categorical type with a specified order
    size_categories = ['<3mm', '3-4mm', '4-5mm', '5-6mm', '6-7mm', '7-8mm', '8-9mm', '9-10mm', '10-11mm', '11-12mm', '>12mm']
    train_df['Size Category'] = pd.Categorical(train_df['Size Category'], categories=size_categories, ordered=True)
    test_df['Size Category'] = pd.Categorical(test_df['Size Category'], categories=size_categories, ordered=True)


    # Aggregate data by size category
    train_counts = train_df.groupby('Size Category')['GT Count'].sum().reset_index()
    test_counts = test_df.groupby('Size Category')['GT Count'].sum().reset_index()

    # Normalize counts to get percentages
    train_total = train_counts['GT Count'].sum()
    test_total = test_counts['GT Count'].sum()

    train_counts['Percentage'] = (train_counts['GT Count'] / train_total) * 100
    test_counts['Percentage'] = (test_counts['GT Count'] / test_total) * 100

    # Merge train and test data for plotting
    merged_df = pd.merge(train_counts, test_counts, on='Size Category', suffixes=('_Train', '_Test'))

    # Plotting the size distribution
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['Size Category'], merged_df['Percentage_Train'], label='Train', linestyle='-', marker='o')
    plt.plot(merged_df['Size Category'], merged_df['Percentage_Test'], label='Test', linestyle='--', marker='x')

    plt.xlabel('Size Category (mm)')
    plt.ylabel('Percentage')
    plt.title('Size Distribution of Aneurysms in Train and Test MR Sets')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save plot
    plt.savefig('size_distribution_MR.png')

    dataset = "CT"

    # Process training files
    train_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTr'
    train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path) if f.endswith('.nii.gz')]
    aggregated_csv_file_name_train = f"{dataset}_count_an_per_size_train.csv"
    count_aneurysms_per_size(train_files, aggregated_csv_file_name_train)
    print(f"Processed {dataset} train ")

    # Process test files
    test_files_path = f'/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset059_IA/labelsTs_internal'
    test_files = [os.path.join(test_files_path, f) for f in os.listdir(test_files_path) if f.endswith('.nii.gz')]
    aggregated_csv_file_name_test = f"{dataset}_count_an_per_size_test.csv"
    count_aneurysms_per_size(test_files, aggregated_csv_file_name_test)
    print(f"Processed {dataset} test ")

    # Plot for CT
    # Load the aggregated data for training and internal test sets
    train_df = pd.read_csv(aggregated_csv_file_name_train)
    test_df = pd.read_csv(aggregated_csv_file_name_test)

    # Set the size category as a categorical type with a specified order
    size_categories = ['<3mm', '3-4mm', '4-5mm', '5-6mm', '6-7mm', '7-8mm', '8-9mm', '9-10mm', '10-11mm', '11-12mm', '>12mm']
    train_df['Size Category'] = pd.Categorical(train_df['Size Category'], categories=size_categories, ordered=True)
    test_df['Size Category'] = pd.Categorical(test_df['Size Category'], categories=size_categories, ordered=True)


    # Aggregate data by size category
    train_counts = train_df.groupby('Size Category')['GT Count'].sum().reset_index()
    test_counts = test_df.groupby('Size Category')['GT Count'].sum().reset_index()

    # Normalize counts to get percentages
    train_total = train_counts['GT Count'].sum()
    test_total = test_counts['GT Count'].sum()

    train_counts['Percentage'] = (train_counts['GT Count'] / train_total) * 100
    test_counts['Percentage'] = (test_counts['GT Count'] / test_total) * 100

    # Merge train and test data for plotting
    merged_df = pd.merge(train_counts, test_counts, on='Size Category', suffixes=('_Train', '_Test'))

    # Plotting the size distribution
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['Size Category'], merged_df['Percentage_Train'], label='Train', linestyle='-', marker='o')
    plt.plot(merged_df['Size Category'], merged_df['Percentage_Test'], label='Test', linestyle='--', marker='x')

    plt.xlabel('Size Category (mm)')
    plt.ylabel('Percentage')
    plt.title('Size Distribution of Aneurysms in Train and Test CT Sets')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save plot
    plt.savefig('size_distribution_CT.png')


