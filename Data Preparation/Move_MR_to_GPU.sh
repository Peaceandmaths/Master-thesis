ssh golubeka@login02.hpc.zhaw.ch
# Copy files from source server to local machine
rsync -aviP golubeka@login02.hpc.zhaw.ch:/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu-lausanne/derivatives/manual_masks "/c/ZHAW/4th semester/Master thesis/Process/Dataset_MR_original"
# Moved files to local machine

echo "Moved files to local machine"

# Copy files from local machine to destination server
rsync -aviP "/c/ZHAW/4th semester/Master thesis/Process/Dataset_MR_original" golubeka@160.85.79.231:/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original

echo "Moved files to the GPU"