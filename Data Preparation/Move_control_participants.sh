#!/bin/bash

# This script moves the brain mask files for the control participants from the source directory to the destination directory.

# Define the source and destination directories
src_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset_MR_original/Images_and_labels"
dest_dir="/data/golubeka/nnUNet_Frame/nnUNet_data/nnUNet_raw/Dataset060_MR"


# Read the participants.txt file and extract the control participants' IDs
control_participants=('sub-000' 'sub-001' 'sub-002' 'sub-005' 'sub-006' 'sub-007' 'sub-015'
 'sub-021' 'sub-030' 'sub-033' 'sub-035' 'sub-036' 'sub-037' 'sub-039'
 'sub-045' 'sub-047' 'sub-053' 'sub-065' 'sub-069' 'sub-073' 'sub-077'
 'sub-078' 'sub-079' 'sub-091' 'sub-092' 'sub-100' 'sub-105' 'sub-107'
 'sub-109' 'sub-111' 'sub-112' 'sub-116' 'sub-119' 'sub-128' 'sub-134'
 'sub-144' 'sub-149' 'sub-151' 'sub-159' 'sub-160' 'sub-162' 'sub-164'
 'sub-165' 'sub-167' 'sub-171' 'sub-172' 'sub-175' 'sub-176' 'sub-178'
 'sub-179' 'sub-183' 'sub-185' 'sub-186' 'sub-187' 'sub-188' 'sub-189'
 'sub-197' 'sub-198' 'sub-202' 'sub-204' 'sub-209' 'sub-214' 'sub-215'
 'sub-222' 'sub-224' 'sub-226' 'sub-228' 'sub-239' 'sub-240' 'sub-242'
 'sub-243' 'sub-245' 'sub-246' 'sub-248' 'sub-250' 'sub-251' 'sub-252'
 'sub-253' 'sub-255' 'sub-256' 'sub-257' 'sub-261' 'sub-262' 'sub-264'
 'sub-265' 'sub-266' 'sub-267' 'sub-268' 'sub-269' 'sub-270' 'sub-271'
 'sub-273' 'sub-275' 'sub-276' 'sub-277' 'sub-278' 'sub-279' 'sub-280'  
 'sub-282' 'sub-284' 'sub-286' 'sub-287' 'sub-288' 'sub-291' 'sub-293'
 'sub-294' 'sub-295' 'sub-297' 'sub-299' 'sub-304' 'sub-306' 'sub-308'
 'sub-316' 'sub-318' 'sub-319' 'sub-324' 'sub-334' 'sub-337' 'sub-339'
 'sub-344' 'sub-345' 'sub-347' 'sub-349' 'sub-351' 'sub-352' 'sub-353'
 'sub-354')


# Move the control participants' files to the destination directory
for id in "${control_participants[@]}"; do
  mv "$src_dir/${id}_ses-"*"_desc-brain_mask.nii.gz" "$dest_dir/"
done