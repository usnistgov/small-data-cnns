# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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
