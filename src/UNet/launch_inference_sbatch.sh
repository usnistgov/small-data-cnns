# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet
#SBATCH -o unet_%N.%j.out
#SBATCH --mail-user=<your_email_here>@nist.gov
#SBATCH --mail-type=FAIL
#SBATCH --time=12:0:0

timestamp="$(date +%Y%m%dT%H%M%S)"
experiment_name="unet-infer-${timestamp}"
echo "Experiment: $experiment_name"

working_dir="/scratch/${SLURM_JOB_ID}"

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
term_handler()
{
        echo "function term_handler called.  Cleaning up and Exiting"
        rm -rf ${working_dir}
        exit -1
}

# associate the function "term_handler" with the TERM signal
trap 'term_handler' TERM

# this is the root directory for results
wrk_directory="/wrk/pnb"

# make working directory
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

#make results directory
results_dir="$wrk_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

# job configuration
checkpoint_filepath=${wrk_directory}/unet-aug-2019-03-01T15\:05\:08/checkpoint/model.ckpt-37
infer_folder="wippModifiedCor"
image_folder=${wrk_directory}/data/${infer_folder}/
number_classes=4
echo "job config: checkpoint_filepath=" $checkpoint_filepath " image_folder=" $image_folder " output_folder=" $results_dir " number_classes=" ${number_classes}

# copy data to node
echo "Copying data to Node"
cp -r ${wrk_directory}/data/${infer_folder}/ ${working_dir}/
echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}

# # **********************
# # Conda setup
# module load anaconda3
# conda create --name tf python=3.6
# conda activate tf

# conda install tensorflow-gpu
# conda install scikit-image
# conda install scikit-learn
# pip install lmdb
# load any modules
module load anaconda3
conda activate tf
echo "Modules loaded"


# launch inference script with required options
echo "Launching Inference Script"
echo "python inference.py --gpu=0 --checkpoint_filepath="$checkpoint_filepath" --image_folder="$image_folder" --output_folder="$results_dir" --number_classes="${number_classes} "| tee "$results_dir"/log.txt"

python inference.py --gpu=0 --checkpoint_filepath="$checkpoint_filepath" --image_folder="$image_folder" --output_folder="$results_dir" --number_classes=${number_classes} | tee "$results_dir/log.txt"

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"
