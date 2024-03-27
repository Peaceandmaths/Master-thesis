import numpy as np
import pandas as pd
from scipy.spatial import distance

""" Here's a Python function that matches aneurysms in predicted files and ground truth masks based on the criteria you specified. 
This function uses the scipy.spatial.distance.cdist function to compute the distances between all pairs of centers, 
and then finds the matches based on the distances and radii. 
It returns a DataFrame with the matched and unmatched aneurysms.
You can call this function with two DataFrames: one for the predicted aneurysms and one for the ground truth aneurysms. 
Each DataFrame should have a 'center_of_mass' column with the centers of the aneurysms and a 'radius' column with the radii of the aneurysms.
Please note that this function assumes that the 'center_of_mass' column contains tuples or lists with the coordinates of the centers, 
and that the 'radius' column contains the radii as numbers. 
If your DataFrames have a different format, you might need to adjust the function accordingly.
Also, this function does not compute the detection metrics. 
You can compute the detection rate as the number of matched predicted aneurysms divided by the total number of ground truth aneurysms, 
and the IoU (Intersection over Union) as the number of matched predicted aneurysms divided by the total number of predicted and ground truth aneurysms minus the number of matches.
 """


# Debugged version : don't give distance_3d nan values + progress bar + concat in stead of append 

import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

# For some reason the coordinates in the center_of_mass column are stored as strings

def literal_eval_nan(x):
    try:
        return ast.literal_eval(x)
    except ValueError:
        return np.nan

# Revised verison of distance_3d to compute all distances between two lists 
    
def distance_3d(list1, list2):
    # Convert the lists to arrays of shape (n, 3)
    array1 = np.array(list(list1))
    array2 = np.array(list(list2))

    # Compute the distances between all pairs of points
    distances = np.linalg.norm(array1[:, np.newaxis] - array2, axis=2)

    return distances


def match_aneurysms(ground_truth_file, predicted_file):
    # Load the data
    ground_truth = pd.read_csv(ground_truth_file)
    predicted = pd.read_csv(predicted_file)

    # Get the unique file names
    file_names = ground_truth['file_name'].unique()

    # Initialize DataFrames to store the matched and unmatched aneurysms using the original column names 
    matched_predicted = pd.DataFrame(columns=predicted.columns)
    matched_ground_truth = pd.DataFrame(columns=ground_truth.columns)
    unmatched_predicted = pd.DataFrame(columns=predicted.columns)
    unmatched_ground_truth = pd.DataFrame(columns=ground_truth.columns)
    true_negatives = pd.DataFrame(columns=predicted.columns)
    # Convert the 'center_of_mass' columns back to tuples
    ground_truth['center_of_mass'] = ground_truth['center_of_mass'].apply(literal_eval_nan)
    predicted['center_of_mass'] = predicted['center_of_mass'].apply(literal_eval_nan)


    # Iterate over the file names
    for file_name in tqdm(file_names, desc='Matching aneurysms', total = len(file_names)):
        # Filter the aneurysms for the current file
        ground_truth_file = ground_truth[ground_truth['file_name'] == file_name]
        predicted_file = predicted[predicted['file_name'] == file_name]

        # If both predicted and ground truth are nan, it's a true negative
        if pd.isna(predicted_file['center_of_mass']).all() and pd.isna(ground_truth_file['center_of_mass']).all():
            true_negatives = pd.concat([true_negatives, predicted_file])
        else:
            # Remove nan values before computing the distances
            # and reset index
            predicted_file = predicted_file.dropna(subset=['center_of_mass']).reset_index(drop=True)
            ground_truth_file = ground_truth_file.dropna(subset=['center_of_mass']).reset_index(drop=True)

            # If either DataFrame is empty, skip this iteration
            if predicted_file.empty or ground_truth_file.empty:
                continue
           
            # Compute the distances between all pairs of centers
            distances = distance_3d(predicted_file['center_of_mass'], ground_truth_file['center_of_mass'])

            # Iterate over the predicted aneurysms
            for i in range(len(predicted_file)):
                # Iterate over the ground truth aneurysms
                for j in range(len(ground_truth_file)):
                    # If the center distance is smaller than the summation of the radii
                    if distances[i, j] < (predicted_file['radius_voxels'][i] + ground_truth_file['radius_voxels'][j]):
                        # Mark them as matched
                        matched_predicted = pd.concat([matched_predicted, predicted_file.iloc[i].to_frame().T])
                        matched_ground_truth = pd.concat([matched_ground_truth, ground_truth_file.iloc[j].to_frame().T])
                        # iou overlap 0.5, 0.1 
                        
    # Find the unmatched aneurysms
    unmatched_predicted = predicted.drop(matched_predicted.index)
    unmatched_ground_truth = ground_truth.drop(matched_ground_truth.index)

    # Save the matched and unmatched aneurysms to CSV files
    matched_predicted.to_csv('matched_predicted_2000_f0.csv', index=False)
    matched_ground_truth.to_csv('matched_ground_truth_2000_f0.csv', index=False)
    unmatched_predicted.to_csv('unmatched_predicted_2000_f0.csv', index=False)
    unmatched_ground_truth.to_csv('unmatched_ground_truth_2000_f0.csv', index=False)
    true_negatives.to_csv('true_negatives_2000_f0.csv', index=False)


# Call the function

ground_truth_file = 'stats_internal_gt.csv'
predicted_file = 'stats_internal_2000_f0.csv'
match_aneurysms(ground_truth_file, predicted_file)

# Checking distance function 

import pandas as pd
import numpy as np
from ast import literal_eval

def literal_eval_nan(v):
    try:
        return literal_eval(v)
    except ValueError:
        return np.nan

# Revise distance_3d function 
    
def distance_3d(list1, list2):
    # Convert the lists to arrays of shape (n, 3)
    array1 = np.array([np.array(x) for x in list1])
    array2 = np.array([np.array(x) for x in list2])

    # Compute the distances between all pairs of points
    distances = np.array([[np.linalg.norm(a - b) for b in array2] for a in array1])

    return distances

# Load the data
ground_truth = pd.read_csv('stats_internal_gt.csv')
predicted = pd.read_csv('stats_internal_predicted.csv')

# Convert the 'center_of_mass' columns back to tuples
ground_truth['center_of_mass'] = ground_truth['center_of_mass'].apply(literal_eval_nan)
predicted['center_of_mass'] = predicted['center_of_mass'].apply(literal_eval_nan)

# Filter the aneurysms for the file 'Ts_0017'
ground_truth_file = ground_truth[ground_truth['file_name'] == 'Ts_0017']
predicted_file = predicted[predicted['file_name'] == 'Ts_0017']
print(ground_truth_file, 'ground_truth_file')
print(predicted_file, 'predicted_file')
# Remove nan values
ground_truth_file = ground_truth_file.dropna(subset=['center_of_mass'])
predicted_file = predicted_file.dropna(subset=['center_of_mass'])
# Compute the distances between all pairs of centers
distances = distance_3d(predicted_file['center_of_mass'], ground_truth_file['center_of_mass'])
# Print the distances
print(distances)


# Getting detection rate per aneurysm size

def detection_rate(matched_file, ground_truth_file): 
    """ Loading the csv file, counting the number of rows and dividing by the number of rows in the ground truth file for 3 groups of aneurysm size : 
     < 3mm of size_mm 
      between 3 and 7 mm of size_mm
      > 7 mm of size_mm
      and calculating detetcion rate for each group

      """
    # Load the data
    matched = pd.read_csv(matched_file).drop_duplicates()
    ground_truth = pd.read_csv(ground_truth_file)

    # Count the number of ground truth aneurysms
    # from stats_internal_gt.csv compute total number of anuerysms per size
    n_gt_less5 = len(ground_truth[ground_truth['size_mm'] < 5])
    n_gt_5_7 = len(ground_truth[(ground_truth['size_mm'] >= 5) & (ground_truth['size_mm'] < 7)])
    n_gt_more7 = len(ground_truth[ground_truth['size_mm'] > 7])

    # Compute the detection rate per aneurysm size

    detection_rate_less5 = len(matched[matched['size_mm'] < 5]) / n_gt_less5
    detection_rate_5_7 = len(matched[(matched['size_mm'] >= 5) & (matched['size_mm'] < 7)]) / n_gt_5_7
    detection_rate_more7 = len(matched[matched['size_mm'] > 5]) / n_gt_more7

    # Print the detection rate for the current group
    print(f'Detection rate for <5 mm : {detection_rate_less5:.2f}\n'
            f'Detection rate for 5-7 mm : {detection_rate_5_7:.2f}\n'
            f'Detection rate for >7 mm : {detection_rate_more7:.2f}')

    

# call the function
detection_rate('matched_ground_truth_2000_f0.csv', 'stats_internal_gt.csv')

# Average detetcion rate 

def average_detection_rate(matched_file, ground_truth_file):
    """ Detected 286 aneurysms in the internal test set overall"""
    # Load the data
    matched = pd.read_csv(matched_file).drop_duplicates()
    ground_truth = pd.read_csv(ground_truth_file)

    # Count the number of ground truth aneurysms
    n_gt = len(ground_truth)
    print(n_gt, 'n_gt')
    print(len(matched), 'len(matched)')

    # Compute the detection rate
    detection_rate = len(matched) / n_gt

    # Print the detection rate
    print(f'Average detection rate: {detection_rate:.2f}')

# Call the function
average_detection_rate('matched_ground_truth_2000_f0.csv', 'stats_internal_gt.csv')


import pandas as pd

# Load the data
df = pd.read_csv('matched_ground_truth.csv')

# Count the number of unique rows
# For some reason I get duplicates in the matched_ground_truth.csv and matched_predicted.csv files
n_unique_rows = df.drop_duplicates().shape[0]

# Print the number of unique rows
print(f'Number of unique rows: {n_unique_rows}')
