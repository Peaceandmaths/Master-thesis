# Master-thesis : Comparative Evaluation of Deep Learning Networks for Intracranial Aneurysm Segmentation

This repository contains the code and data processing pipeline for the master thesis project.  
The project involves the preparation, training and evaluation of the nn-UNet segmentation model 

__________________________________________________________________________________________________

## Table of Contents

1. [Requirements](#requirements)
2. [Data Exploration](#data-exploration)
3. [Data Preparation](#data-preparation)
4. [Total Segmentator](#total-segmentator)
5. [Model Training](#model-training)
6. [Model Prediction](#model-prediction)
7. [Post-processing](#post-processing)
8. [Evaluation](#evaluation)
9. [Running the Workflow](#running-the-workflow)

## Requirements

- Python 3.12.4
- Required Python packages (list in `requirements.txt`)
- nnU-Net version 2.2.

## Data Exploration 

- [Check_image_label_shapes](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Exploration/Check_image_label_shapes.py) checks that each image has a corresponding label of the same shape. For the CT dataset one of the images didn't have a label of the same size, so it was discarded.
- [Checking_image_resolution](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Exploration/Checking_image_resolution.py) categorizes images from the CT dataset into groups of different resolution (in terms of voxel size).
- [count_an_per_size_trainset](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Exploration/count_an_per_size_trainset.py) counts aneurysm sizes in the original train dataset in different size categories, with my definition of aneurysm size.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5d8f5f7-d770-4242-b961-1dc02771cb34" alt="Image Description">
</p>

## Data Preparation

- To use NN-UNet, [generate_dataset_json_file](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/generate_dataset_json_file.py) and [dataset_conversion_file](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/dataset_conversion_file.py) were used to create the expected folder structure and to follow the NN-UNet naming conventions. 

### CT dataset (id = 059)
Flowchart for the CT data preparation and nnunet commands ( without TotalSegmentator ) 

![image](https://github.com/Peaceandmaths/Master-thesis/assets/117741432/a7f3aa2a-2c49-476b-9c8c-379a2918eecd)


 - CT raw dataset was collected from open source Zendo ( see reference), including training images (imagesTr, n = 1186)) and test images (imagesTs_internal, n= 152)).
- The whole dataset was copied to perform further modifications on the copy ( renaming, preprocessing etc).
-  Rename data to nnU-Net format and store old and new names in a lookup table, running nnunet commands like verify data integrity, preprocessing and train. All these steps are described [here](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/Data_version_control_latest.py).
-  [Rename the test set](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/renaming_test_files.py) too ( it was named differently from the train data).

Flowchart fro the CT data processing with Total Segmentator 

![image](https://github.com/user-attachments/assets/02b2bd86-5da9-4035-b75a-5760e3a1d255)



### MR dataset (id = 060)

- Originally, in the MR dataset available online, only patients with aneurysms had corresponding label files, the control participants didn't have label files. I [created empty images](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/Creating_empty_labels_MR.py) with the same dimensions to provide nnunet with images and labels files in pairs.
- For patients with multiple aneurysms, there were multiple label files with one aneurysm per each label file. I [merged](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/Merge_lesions_MR.py) multiple lesion in the same label files so each aptient has only one corresponding label.
- Files should have been [renamed](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/renaming_files_nnUnet_format.py) to nnunet format.
- Split all training files into validation and test [here](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Preparation/train_val_test_MR_manual_split.py). 
- The MR images had been pre-cropped to encompass the brain and the specific region of interest within the brain (Circle of Willis). Consequently, the MR dataset did not require processing through the Total Segmentator step, unlike the CT dataset.
![image](https://github.com/user-attachments/assets/ad85cc04-7e96-426c-9a0d-01d408845615)


## Total Segmentator

- See instructions to use TotalSegmentator [here](https://github.com/wasserth/TotalSegmentator).
- The [bash script](https://github.com/Peaceandmaths/Master-thesis/blob/main/Total%20Segmentator/total_segmentator_10test.sh) to run TS over multiple images in a folder
- Once the brain masks are available, we [crop](https://github.com/Peaceandmaths/Master-thesis/blob/main/Total%20Segmentator/cropping_script.py) the original images to the brain region and add padding.
- Warning: check the volume of the label before and after cropping, TS shouldn't cut out labels inside the brain. But it is possible to see aneurysms outside of the brain.

![image](https://github.com/user-attachments/assets/413fc9af-e1d8-4d1b-9340-2a386416073c)


## Model Training

NN-UNet model training consists of runnnig the following NN-UNet commands consecutively : 
1) `nnUNetv2_plan_and_preprocess` and `--verify_dataset_integrity` - Generate dataset fingerprint and plans 
2) `nnUNetv2_train` - Train the models using the 5-fold cross-validation approach, choose `3d_fullres`
3) `nnUNetv2_find_best_configuration`- Find the best configuration for the dataset using the specified folds.
4) `nnUNetv2_predict` - Generate predictions for the internal test set.
5) `nnUNetv2_apply_postprocessing` - Apply post-processing to the predictions to refine results.

[This bash script](https://github.com/Peaceandmaths/Master-thesis/blob/main/NN-Unet%20pipeline/preprocess_train_find_best_060.sh) runs all these steps automatically from step 1 to 3. Steps from 3 till 5 are automated [here](NN-Unet pipeline/script_060_4.sh). There's a break at step 3 because it needs user intervention. You have to see generated instructions and run the code as specified. 


The post-processing is done by the default nnunet procedure, the code can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/postprocessing/remove_connected_components.pY), remove all but the largest component code is [here](https://github.com/MIC-DKFZ/acvl_utils/blob/master/acvl_utils/morphology/morphology_helper.py#L33). The way it works is that it first does a connected component analysis, which also gives the size per component and then filters for the largest component. An additional post-processing step is removing connected components smaller than 1mm in diameter. This is done in the evaluation script (see below). 


## Evaluation

NN-UNet generates an evaluation summary that gives some basic metrics across all images on the vowel-level. 
I implemented my own evaluation pipeline automated [here](Evaluation/all_evaluation_metrics.sh). 

![image](https://github.com/user-attachments/assets/b3eb469b-420a-479c-9b06-c4ca7d84bea7)


## References : 

- nnUnet : Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, https://doi.org/10.1038/s41592-020-01008-z .

- TotalSegmentaor : Akinci D'Antonoli, T., Berger, L. K., Indrakanti, A. K., Vishwanathan, N., Weiß, J., Jung, M., Berkarda, Z., Rau, A., Reisert, M., Küstner, T., Walter, A., Merkle, E. M., Segeroth, M., Cyriac, J., Yang, S., & Wasserthal, J. (2024). TotalSegmentator MRI: Sequence-Independent Segmentation of 59 Anatomical Structures in MR Images. arXiv preprint arXiv:2405.19492. https://doi.org/10.48550/arXiv.2405.19492.

- CT dataset : Z.-H. Bo. Large IA Segmentation dataset. Version 1. This dataset is for non-commercial
purposes. Feb. 2021. DOI: 10 . 5281 / zenodo . 6801398. URL: https://zenodo.org/record/6801398 .

- MR dataset : T. Di Noto et al. “Towards Automated Brain Aneurysm Detection in TOFMRA: Open Data, Weak Labels, and Anatomical Knowledge”. In: Neuroinformatics 21.1 (2023), pp. 21–34. ISSN: 1559-0089. DOI: 10.1007/s12021-022-
09597-0. URL: https://doi.org/10.1007/s12021-022-09597-0.

