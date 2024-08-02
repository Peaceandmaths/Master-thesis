from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from generate_dataset_json_file import generate_dataset_json
import nnunetv2
from nnunetv2 import paths
#from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed

# Run setting env variables in terminal before executing
# export nnUNet_raw="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"  
# export nnUNet_results="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"
# export nnUNet_preprocessed="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"

nnUNet_raw = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"
nnUNet_preprocessed = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"
nnUNet_results = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"


nnUNet_raw = nnunetv2.paths.nnUNet_raw
nnUNet_preprocessed = nnunetv2.paths.nnUNet_preprocessed
nnUNet_results = nnunetv2.paths.nnUNet_results


def convert_largeia(largeia_base_dir:str = '/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA/',
                     nnunet_dataset_id: int = 230):
    task_name = "IA"

    foldername = "Dataset%s_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # paths to the original images and labels
    orig_images_dir = join(largeia_base_dir, "imagesTr")
    orig_labels_dir = join(largeia_base_dir, "labelsTr")

   # copy images
    image_files = [f for f in os.listdir(orig_images_dir) if f.startswith('T') and f.endswith('.nii.gz')]
    identifiers = []
    for image_file in image_files:
        identifier = os.path.splitext(image_file)[0]
        identifiers.append(identifier)
        if not isfile(join(imagestr, f'{identifier}.nii.gz')):
            shutil.copy(join(orig_images_dir, image_file), join(imagestr, image_file))
    
    # copy labels
    label_files = [f for f in os.listdir(orig_labels_dir) if f.startswith('T') and f.endswith('.nii.gz')]
    for label_file in label_files:
        identifier = os.path.splitext(label_file)[0]
        if not isfile(join(labelstr, f'{identifier}.nii.gz')):
            shutil.copy(join(orig_labels_dir, label_file), join(labelstr, label_file))



# check which if statements to leave in, and which to take out
    # create identifiers list from image_files and label_files
    identifiers = [os.path.splitext(f)[0] for f in image_files + label_files]
    train_images = [i for i in identifiers if i.startswith('Tr')]
    val_images = [i for i in identifiers if i.startswith('Ts')]
    n = len(train_images)
    print(n)

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "aneurysm": 1
                          },
                          num_training_cases= n, file_ending='.nii.gz',
                          dataset_name=task_name, reference='https://doi.org/10.5281/zenodo.6801398',
                          release='12/12/2021',
                          # overwrite_image_reader_writer='NibabelIOWithReorient',
                          description=task_name)

    # manual split 
    ######################################## I can split differently if I want to ########################################

    
    splits = [{'train': train_images, 'val': val_images}]

    pp_out_dir = join(nnUNet_preprocessed, foldername)
    maybe_mkdir_p(pp_out_dir)
    save_json(splits, join(pp_out_dir, 'splits_final.json'), sort_keys=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted largeia dataset (must have Tr_XXXX subfolders)")
    parser.add_argument('-d', required=False, type=int, default=230, help='nnU-Net Dataset ID, default: 230')
    args = parser.parse_args()
    amos_base = args.input_folder
    convert_largeia(amos_base, args.d)