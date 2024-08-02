import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths for the images and labels
images_folder_path = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr"
labels_folder_path = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr"

# Define the paths for the test sets
images_test_folder_path = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTs"
labels_test_folder_path = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTs"

# Create the test directories if they do not exist
os.makedirs(images_test_folder_path, exist_ok=True)
os.makedirs(labels_test_folder_path, exist_ok=True)

# Get the filenames
filenames = sorted(os.listdir(images_folder_path))

# Split the data into training and testing sets
train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, random_state=42)

# Move the test files to the test folders
for filename in test_filenames:
    # Extract the ID from the image filename
    id = filename.split('_')[1]
    # Construct the label filename from the ID
    label_filename = f"Tr_{id}.nii.gz"
    shutil.move(f"{images_folder_path}/{filename}", f"{images_test_folder_path}/{filename}")
    shutil.move(f"{labels_folder_path}/{label_filename}", f"{labels_test_folder_path}/{label_filename}")