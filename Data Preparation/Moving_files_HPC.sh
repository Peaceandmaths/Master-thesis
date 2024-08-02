#!/bin/bash

ssh golubeka@login02.hpc.zhaw.ch

rsync -aviP /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_internal golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/golubeka/nnUnet_raw/Dataset057_IA
rsync -aviP /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_internal golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/golubeka/nnUnet_raw/Dataset057_IA

rsync -aviP /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/labelsTs_external golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/golubeka/nnUnet_raw/Dataset057_IA
rsync -aviP /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset057_IA/imagesTs_external golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/golubeka/nnUnet_raw/Dataset057_IA


rsync -aviP /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset059_IA golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/golubeka/nnUnet_results

# Left to copy : Dataset057_IA
rsync -aviP /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results/Dataset057_IA golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/golubeka/nnUnet_results

# MR data is at /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu-lausanne

# To copy the data from the HPC to the GPU 

mkdir -p /data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original
rsync -aviP golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu-lausanne/derivatives/manual_masks golubeka@160.85.79.231:/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original

################ I can't use rsync between both remote machines, I need to pass by my local machine 

# Copy files from source server to local machine
rsync -aviP golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu-lausanne/derivatives/manual_masks C:\ZHAW\4th semester\Master thesis\Process\Dataset_MR_original

# Moved files to local machine

echo "Moved files to local machine"

# Copy files from local machine to destination server
rsync -aviP C:\ZHAW\4th semester\Master thesis\Process\Dataset_MR_original golubeka@160.85.79.231:/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original

echo "Moved files to the GPU"