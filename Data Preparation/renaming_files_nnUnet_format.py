import os 

def rename_files_new_format():
    images_folder_path = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/imagesTr"
    labels_folder_path = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR/labelsTr"

    # Make sure to sort both lists in the same order
    images_filenames = sorted(os.listdir(images_folder_path))
    labels_filenames = sorted(os.listdir(labels_folder_path))

    for count, (img_filename, lbl_filename) in enumerate(zip(images_filenames, labels_filenames)):
        # Extract the id from the filename
        id = img_filename.split('_')[0][4:]

        # Construct the new filenames
        img_dst = f"Tr_{id}_0000.nii.gz"
        lbl_dst = f"Tr_{id}.nii.gz"

        # Construct the full paths for the source and destination files
        img_src = f"{images_folder_path}/{img_filename}"
        img_dst = f"{images_folder_path}/{img_dst}"
        lbl_src = f"{labels_folder_path}/{lbl_filename}"
        lbl_dst = f"{labels_folder_path}/{lbl_dst}"

        # Rename the files
        os.rename(img_src, img_dst)
        os.rename(lbl_src, lbl_dst)

# Call the function
rename_files_new_format()