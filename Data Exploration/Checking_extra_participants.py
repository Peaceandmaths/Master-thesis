# Read the particiapnts.tsv file, extract the unique list of participant_id and count them 
# do the same for the images in the imagesTr directory 
# compare the ids and their number 

import pandas as pd
import os

# Read the participants.tsv file
participants_tsv = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/participants.txt'
participants = pd.read_csv(participants_tsv, sep='\t')

# Extract the unique list of participant_id and count them
unique_participants = participants['participant_id'].unique()
num_unique_participants = len(unique_participants)

# number of participants in the metadata = 284

# Count the images in the imagesTr directory
image_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
# Extract the unique IDs from the image filenames
image_ids = {f.split('_')[0] for f in image_files if f.startswith('sub-')}
num_image_ids = len(image_ids)

# Compare the ids and their number
print(f'Number of unique participants: {num_unique_participants}')
print(f'Number of unique ids in image folder: {image_ids}')

if num_unique_participants == num_image_ids:
    print('The number of unique participants matches the number of images.')
else:
    print('The number of unique participants does not match the number of images.')



# Extract the control patients id to create empty masks 

# Read the participants.txt file
participants_txt = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/participants.txt'
participants = pd.read_csv(participants_txt, sep='\t')

# Extract the control patients id
control_participants = participants[participants['group'] == 'control']['participant_id'].unique()

print(f'Control participants: {control_participants}')
print(f'Number of control participants: {len(control_participants)}')


# Checking the copied control participants 


# Count the images in the imagesTr directory
image_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
# Extract the unique IDs from the image filenames
image_ids = {f.split('_')[0] for f in image_files if f.startswith('sub-')}
num_image_ids = len(image_ids)
len(image_files)

# One healthy participant has 2 sessions 


################################# 

# Listing duplicate sessions 

# Read the participants.tsv 
import pandas as pd
participants_tsv = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/participants.txt'
participants = pd.read_csv(participants_tsv, sep='\t')

duplicates = participants[participants.duplicated(['participant_id'], keep=False)]
print(duplicates)
print(len(duplicates))

duplicate_ids=('sub-074_ses-20100421', 
               'sub-170_ses-20111112',
               'sub-217_ses-20110811',
               'sub-225_ses-20110617',
               'sub-234_ses-20111203',
               'sub-257_ses-20110619',
               'sub-311_ses-20100925',
               'sub-315_ses-20100817', 
               'sub-325_ses-20110127',
               'sub-327_ses-20110124',
               'sub-327_ses-20110923',
               'sub-329_ses-20110126')


# Checking which file is missing 

import os

def extract_patient_ids(directory):
    # List all files in the directory
    filenames = os.listdir(directory)

    # Extract the patient IDs from the filenames
    patient_ids = [f.split('_')[0] for f in filenames]

    return patient_ids

# Specify the directories
dir1 = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr'
dir2 = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr'

# Extract the patient IDs from each directory
labels = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr'
ids_dir1 = set(extract_patient_ids(labels))

import pandas as pd
import os

# Read the participants.tsv file
participants_tsv = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/participants.txt'
participants = pd.read_csv(participants_tsv, sep='\t')

# Extract the unique list of participant_id and count them
unique_participants = set(participants['participant_id'].unique())

#print(ids_dir1)
ids_dir2 = extract_patient_ids(dir2)
print(ids_dir2)
# Find the IDs that are in dir1 but not in dir2
missing_in_dir2 = ids_dir1 - unique_participants

# Find the IDs that are in dir2 but not in dir1
missing_in_dir1 = unique_participants - ids_dir1

# Print the missing IDs
print('IDs in dir1 but not in dir2:', missing_in_dir2)
print('IDs in dir1 but not in dir1:', missing_in_dir1)

len(ids_dir1)
len(ids_dir2)

# Duplicates in label ? 

import pandas as pd
labels = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr'
ret = extract_patient_ids(labels)
print(len(ret))

duplicates = labels[labels.duplicated([''], keep=False)]
print(duplicates)
print(len(duplicates))

# Find duplicates in a list of strings 

def find_duplicates(lst):
    # Create an empty set to store unique elements
    unique = set()
    duplicates = set()

    # Iterate over the list
    for item in lst:
        # If the item is already in the unique set, it's a duplicate
        if item in unique:
            duplicates.add(item)
        else:
            unique.add(item)

    # Return the duplicates
    return list(duplicates)

# Test the function
ret = extract_patient_ids(labels)
duplicates = find_duplicates(ret)
print('Duplicates:', duplicates)


# missing one 
# Find the row for the participant with id 'sub-482'
participant_row = participants.loc[participants['participant_id'] == 'sub-482']

# Print the row
print(participant_row)