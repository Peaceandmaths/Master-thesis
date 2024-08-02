#!/bin/bash
set -e
# Set environment variables
export nnUNet_raw="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw"  
export nnUNet_results="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_results"
export nnUNet_preprocessed="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_preprocessed"

bash script_060_0.sh
bash script_060_1.sh
bash script_060_2.sh
bash script_060_3.sh
bash script_060_4.sh