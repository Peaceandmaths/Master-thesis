import numpy as np
import pandas as pd
import nibabel as nb
from pathlib import Path

def collect_mask_info(path):
    path = Path(path)
    files = sorted(path.glob("*.nii.gz"))
    data = []
    for i,f in enumerate(files):
        print("%3d/%3d (%.1f%%)" % (i+1, len(files), (i+1)/len(files)*100),
              flush=True)
        info = {}
        mask = nb.load(f)
        header = mask.header
        info["name"] = f.stem
        info["dimX"] = header["dim"][1]
        info["dimY"] = header["dim"][2]
        info["dimZ"] = header["dim"][3]
        info["resX"] = round(header["pixdim"][1], 3)
        info["resY"] = round(header["pixdim"][2], 3)
        info["resZ"] = round(header["pixdim"][3], 3)
        info["nVox"] = np.prod(header["dim"][1:4])
        info["nMask"] = int(mask.get_fdata().sum())
        data.append(info)
    
    
    df = pd.DataFrame(data)
    df.to_csv(path.name+"_055.csv")

path_images_original = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/imagesTs_internal"
path_labels_original = "/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset055_IA/labelsTs_internal/"

collect_mask_info(path_images_original)
collect_mask_info(path_labels_original)


import pandas as pd

# Comparing the two csv files on the dimensions columns 

# Load the CSV files into pandas DataFrames
df_labels = pd.read_csv("labelsTs_internal_055.csv")
df_images = pd.read_csv("imagesTs_internal_055.csv")

# Merge the two DataFrames on the "name" column
df = pd.merge(df_labels, df_images, on="name")
df.head
# Compare the "dimX", "dimY", and "dimZ" columns
mask = (df["dimX_x"] != df["dimX_y"]) | (df["dimY_x"] != df["dimY_y"]) | (df["dimZ_x"] != df["dimZ_y"])

# Print out the names of the images where these dimensions do not match
for name in df[mask]["name"]:
    print(f"The dimensions of the image '{name}' do not match.")
