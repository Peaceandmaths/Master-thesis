# Master-thesis : Comparative Evaluation of Deep Learning Networks for Intracranial Aneurysm Segmentation

This repository contains the code and data processing pipeline for the master thesis project.  
The project involves the preparation, training and evaluation of the nn-UNet segmentation model 

__________________________________________________________________________________________________

## Table of Contents

1. [Requirements](#requirements)
2. [Data Exploration](#data-exploration)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Model Prediction](#model-prediction)
6. [Post-processing](#post-processing)
7. [Evaluation](#evaluation)
8. [Running the Workflow](#running-the-workflow)

## Requirements

- Python 3.12.4
- Required Python packages (list in `requirements.txt`)
- nnU-Net version 2.2.

## Data Exploration 

- [Check_image_label_shapes](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Exploration/Check_image_label_shapes.py) check that each image has a corresponding label of the same shape. For the CT dataset one of the images didn't have a label of the same size, so it was discarded.
- [Checking_image_resolution](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Exploration/Checking_image_resolution.py) categorizes images from the CT dataset into groups of different resolution (in terms of voxel size).
- [count_an_per_size_trainset](https://github.com/Peaceandmaths/Master-thesis/blob/main/Data%20Exploration/count_an_per_size_trainset.py) counts aneurysm sizes in the original train dataset in different size categories, with my definition of aneurysm size.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5d8f5f7-d770-4242-b961-1dc02771cb34" alt="Image Description">
</p>

## Data Preparation

Flowchart for the CT data preparation and nnunet commands ( without TotalSegmentator ) 

<p align="center">
![image](https://github.com/Peaceandmaths/Master-thesis/assets/117741432/a7f3aa2a-2c49-476b-9c8c-379a2918eecd)
</p>

1. **Raw Data Collection**:
   - **Files**: Dataset001_IA
   - **Description**: Collect raw data including training images (imagesTr) and test images (imagesTs).

2. **Copy Data**:
   - **Files**: `Dataset057_IA`
   - **Description**: Create a copy of the raw dataset to `Dataset057_IA`.

3. **Rename Data**:
   - **Files**: `Dataset057_IA`
   - **Description**: Rename data to nnU-Net format and store old and new names in a lookup table.

4. **Plan and Preprocess**:
   - **Script**: `nnunetv2_plan_and_preprocess`
   - **Input**: `Dataset057_IA`
   - **Description**: Preprocess the dataset to create a fingerprint, split into 5 folds, and generate nnU-Net plans and ground truth segmentations.

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

