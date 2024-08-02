import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Check the coordinates code, does it save a rectangular bouding box or the shape of the brain?

input_im_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTr'
output_im_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_cropped/imagesTr'
mask_im_dir = 'data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_brain/imagesTr'

input_lab_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/labelsTr'
output_lab_dir = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_cropped/labelsTr'

# Get the list of files
files = os.listdir(input_im_dir)

for filename in tqdm(files, desc = "Cropping files", total = len(files)):
    if filename.endswith('.nii.gz'):
        unique_id = filename[3:7]  # Extract the unique identifier

        # Process the image
        image = nib.load(os.path.join(input_im_dir,  'Tr_' + unique_id + '_0000.nii.gz'))
        mask = nib.load(os.path.join(mask_im_dir, 'Tr_brain_' + unique_id + '.nii.gz'))

        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        # Set all voxels outside of the mask to 0
        image_data[mask_data == 0] = 0

        # Get the coordinates of the brain region
        coords = np.where(mask_data)
        # check the coordinates type (bounding box or weird shape, rectangular)

        # Crop the image to the brain region
        cropped_image_data = image_data[coords]


        # Create a new Nifti1Image with the cropped data
        cropped_image = nib.Nifti1Image(cropped_image_data, image.affine, image.header)

        # Save the cropped image
        nib.save(cropped_image, os.path.join(output_im_dir, 'Tr_cropped_' + unique_id + '_0000.nii.gz'))

        # Process the label
        label = nib.load(os.path.join(input_lab_dir, 'Tr_' + unique_id + '.nii.gz'))

        label_data = label.get_fdata()

        # Crop the label to the brain region
        cropped_label_data = label_data[coords]

        # Create a new Nifti1Image with the cropped data
        cropped_label = nib.Nifti1Image(cropped_label_data, label.affine, label.header)

        # Save the cropped label
        nib.save(cropped_label, os.path.join(output_lab_dir, 'Tr_cropped_' + unique_id + '.nii.gz'))

        # Warning : check the label(gt) volume after the segmentation (should be the same) 
        # sum voxels = volume 
