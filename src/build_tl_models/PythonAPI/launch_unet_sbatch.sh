# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=unet_tl
#SBATCH -o unet_tl%N.%j.out
#SBATCH --time=48:0:0


wrk_directory="/wrk/mmajursk/small-data-cnns/data/"
source_code_directory="/wrk/mmajursk/small-data-cnns/cocoapi/PythonAPI/"

# job configuration
epoch_count=50
batch_size=8 # 4x across the gpus

# make working directory
working_dir="/scratch/mmajursk/"
rm -rf ${working_dir}
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

# copy data to node
echo "Copying data to Node"
cd ${working_dir}

fn="annotations_trainval2017.zip"
src="$wrk_directory$fn"
cp ${src} ${working_dir}
unzip ${working_dir}/${fn} > /dev/null

fn="val2017.zip"
src="$wrk_directory$fn"
cp ${src} ${working_dir}
unzip ${working_dir}/${fn} > /dev/null

fn="train2017.zip"
src="$wrk_directory$fn"
cp ${src} ${working_dir}
unzip ${working_dir}/${fn} > /dev/null

echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}

# load any modules
module load powerAI/tensorflow
echo "Modules loaded"

results_dir="$wrk_directory/model-tl-unet"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

# launch training script with required options
echo "Launching Training Script"
cd ${source_code_directory}
python ${source_code_directory}/train_unet.py  --batch_size=${batch_size} --epoch_count=${epoch_count} --data_dir=${working_dir} --output_dir="$results_dir"

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"