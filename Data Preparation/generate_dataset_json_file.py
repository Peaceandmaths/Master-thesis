from typing import Tuple

from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):

    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didnt like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
# Generates a dataset.json file in the output folder

output_folder = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset001_IA"
channel_names = {
        0: 'CT'
    }
'''channel_names:
    Channel names must map the index to the name of the channel, example:
    {
        0: 'T1',
        1: 'CT'
    }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.
'''
labels = {
        'background': 0,
        'aneurysm': 1
    }
""" 
labels:
    This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
    Example regular labels:
    {
        'background': 0,
        'left atrium': 1,
        'some other label': 2
    }
    Example region-based training:
    {
        'background': 0,
        'whole tumor': (1, 2, 3),
        'tumor core': (2, 3),
        'enhancing tumor': 3
    } 
    Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
"""
num_training_cases = 1186 
# num_training_cases: is used to double check all cases are there!

file_ending: str = '.nii.gz'
# file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
# segmentations! 
# is it a text file with all the file endings ? 

regions_class_order: Tuple[int, ...] = None,
dataset_name: str = 'Large IA Segmentation'
reference: str = 'https://doi.org/10.5281/zenodo.6801398'
release: str = '12/12/2021'
license: str = 'CC-BY-NC-SA-4.0',
description: str = 'Intracranial Aneurysm Segmentation'
'''dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
completeness and as a reminder that these would be great!'''

overwrite_image_reader_writer: str = None
# overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
# BaseReaderWriter, place it into nnunet.imageio and reference it here by name

generate_dataset_json(output_folder, channel_names, labels, num_training_cases, file_ending,
                      regions_class_order, dataset_name, reference, release, license, description,
                      overwrite_image_reader_writer)