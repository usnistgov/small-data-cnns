#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --job-name=segnet_tl
#SBATCH -o segnet_tl%N.%j.out


wrk_directory="/wrk/mmajursk/small-data-cnns/data/"
source_code_directory="/wrk/mmajursk/small-data-cnns/cocoapi/PythonAPI/"

# job configuration
epoch_count=50
batch_size=4 # 4x across the gpus

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

results_dir="$wrk_directory/model-tl-segnet"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

# launch training script with required options
echo "Launching Training Script"
cd ${source_code_directory}
python ${source_code_directory}/train_segnet.py --batch_size=${batch_size} --epoch_count=${epoch_count} --data_dir=${working_dir} --output_dir="$results_dir"

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"