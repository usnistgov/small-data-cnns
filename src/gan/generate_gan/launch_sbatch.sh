#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=gan
#SBATCH -o %N.%j.out

# job configuration
dataset=${1}

wrk_directory="/wrk/mmajursk/small-data-cnns/data/"
source_code_directory="/wrk/mmajursk/small-data-cnns/src/gan_creation/"


# make working directory
working_dir="/scratch/mmajursk/$dataset/gan-creation/"
rm -rf ${working_dir}
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

# copy data to node
echo "Copying data to Node"
cd ${working_dir}

fn="$dataset.zip"
src="$wrk_directory$fn"
cp ${src} ${working_dir}
unzip ${working_dir}/${fn} > /dev/null
echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}

# load any modules
module load powerAI/tensorflow
echo "Modules loaded"

# launch training script with required options
echo "Launching Training Script"
cd ${source_code_directory}
python main.py --dataset="$dataset" --data_dir=${working_dir} --input_fname_pattern="*.tif" --train
echo "Python script terminated"

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"