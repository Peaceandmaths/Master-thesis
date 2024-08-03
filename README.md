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




## Model Training

1. **Find Best Configuration**:
   - **Script**: `nnUNetv2_find_best_configuration`
   - **Input**: `Dataset057_IA`
   - **Description**: Find the best configuration for the dataset using the specified folds.

2. **Train Models**:
   - **Script**: `nnUNetv2_train`
   - **Input**: `Dataset057_IA`
   - **Description**: Train the models using the 5-fold cross-validation approach.

## Model Prediction

1. **Predict**:
   - **Script**: `nnUNetv2_predict`
   - **Input**: `Dataset057_IA`
   - **Description**: Generate predictions for the internal test set.

## Post-processing

1. **Apply Post-processing**:
   - **Script**: `nnUNetv2_apply_postprocessing`
   - **Input**: `Dataset057_IA`
   - **Description**: Apply post-processing to the predictions to refine results.

## Evaluation

1. **Evaluate Models**:
   - **Script**: `nnUNetv2_evaluate`
   - **Input**: `Dataset057_IA`
   - **Description**: Evaluate the model performance using various metrics.

## Running the Workflow

To reproduce the workflow described in this repository, follow the steps below:

1. **Step 1: Data Preparation**:
   - Collect raw data and store it in the specified format.
   - Run the `Data_version_control_latest.py` script to prepare the dataset.
   - Execute the `nnunetv2_plan_and_preprocess` command to preprocess the data.

2. **Step 2: Model Training**:
   - Run `nnUNetv2_find_best_configuration` to identify the best settings for your model.
   - Execute `nnUNetv2_train` to train the model using the preprocessed data.

3. **Step 3: Model Prediction**:
   - Use the `nnUNetv2_predict` script to generate predictions on the test set.

4. **Step 4: Post-processing**:
   - Apply post-processing using `nnUNetv2_apply_postprocessing` to refine your predictions.

5. **Step 5: Evaluation**:
   - Finally, run the `nnUNetv2_evaluate` script to evaluate the performance of your models using various metrics.

## References : 

- nnUnet : Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, https://doi.org/10.1038/s41592-020-01008-z .

- CT dataset : Z.-H. Bo. Large IA Segmentation dataset. Version 1. This dataset is for non-commercial
purposes. Feb. 2021. DOI: 10 . 5281 / zenodo . 6801398. URL: https://zenodo.org/record/6801398 .

- MR dataset : T. Di Noto et al. “Towards Automated Brain Aneurysm Detection in TOFMRA: Open Data, Weak Labels, and Anatomical Knowledge”. In: Neuroinformatics 21.1 (2023), pp. 21–34. ISSN: 1559-0089. DOI: 10.1007/s12021-022-
09597-0. URL: https://doi.org/10.1007/s12021-022-09597-0.

