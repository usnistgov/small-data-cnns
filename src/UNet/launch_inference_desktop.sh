#!/usr/bin/bash


timestamp="$(date +%Y%m%dT%H%M%S)"
experiment_name="unet-aug-infer-${timestamp}"
echo "Experiment: $experiment_name"

# this is the root directory for results
wrk_directory="/home/pnb/raid1/Concrete"

#make results directory
results_dir="$wrk_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

# job configuration
checkpoint_filepath=${wrk_directory}/unet-aug/checkpoint/model.ckpt-37
infer_folder="wippModified"
image_folder=${wrk_directory}/${infer_folder}/
number_classes=4
echo "job config: checkpoint_filepath=" $checkpoint_filepath " image_folder=" $image_folder " output_folder=" $results_dir " number_classes=" ${number_classes}


# launch inference script with required options
echo "Launching Inference Script"
echo "python3 inference.py --gpu=0 --checkpoint_filepath="$checkpoint_filepath" --image_folder="$image_folder" --output_folder="$results_dir" --number_classes="${number_classes} "| tee "$results_dir"/log.txt"

python3 inference.py --gpu=0 --checkpoint_filepath="$checkpoint_filepath" --image_folder="$image_folder" --output_folder="$results_dir" --number_classes=${number_classes} | tee "$results_dir/log.txt"

echo "Job completed"
