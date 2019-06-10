#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=unet-h-a
#SBATCH -o unet-h_%N.%j.out
#SBATCH --mail-user=mmajursk@nist.gov
#SBATCH --mail-type=FAIL
#SBATCH --time=72:0:0


dataset="hes"
rep=3
timestamp="$(date +%Y%m%dT%H%M%S)"
experiment_name="unet-$dataset-${timestamp}"
echo "Experiment: $experiment_name"

working_dir="/scratch/mmajursk/unet/$experiment_name"

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

wrk_directory="/wrk/mmajursk/small-data-cnns/UNet"

# job configuration
test_every_n_steps=1000
batch_size=4 # 4x across the gpus

# make working directory
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

# load any modules
module load powerAI/tensorflow
echo "Modules loaded"

# copy data to node
echo "Copying data to Node"
test_lmdb_file="test-$dataset.lmdb"
cp -r "/wrk/mmajursk/small-data-cnns/data/$test_lmdb_file" ${working_dir}/${test_lmdb_file}


for N in 50 100 200 300 400 500
do
    train_lmdb_file="train-$dataset-$N.lmdb"
    cp -r "/wrk/mmajursk/small-data-cnns/data/$train_lmdb_file" ${working_dir}/${train_lmdb_file}

    results_dir="$wrk_directory/$experiment_name-$N-$rep-noaug"
    if [ ! -d "$results_dir" ]; then
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        echo "Launching Training Script"
        python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" --use_augmentation=0 | tee "$results_dir/log.txt"
    fi


    results_dir="$wrk_directory/$experiment_name-$N-$rep-aug"
    if [ ! -d "$results_dir" ]; then
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        echo "Launching Training Script"
        python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" --use_augmentation=1 | tee "$results_dir/log.txt"
    fi


    results_dir="$wrk_directory/$experiment_name-$N-$rep-tl"
    if [ ! -d "$results_dir" ]; then
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        checkpoint_fp=/wrk/mmajursk/small-data-cnns/data/model-tl-unet/checkpoint/model.ckpt
        echo "Launching Training Script"
        python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" --use_augmentation=0 --restore_checkpoint_filepath="$checkpoint_fp" | tee "$results_dir/log.txt"
    fi

    results_dir="$wrk_directory/$experiment_name-$N-$rep-tl-aug"
    if [ ! -d "$results_dir" ]; then
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        checkpoint_fp="/wrk/mmajursk/small-data-cnns/data/model-tl-unet/checkpoint/model.ckpt"
        echo "Launching Training Script"
        python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" --use_augmentation=1 --restore_checkpoint_filepath="$checkpoint_fp" | tee "$results_dir/log.txt"
    fi

    results_dir="$wrk_directory/$experiment_name-$N-$rep-gan"
    if [ ! -d "$results_dir" ]; then
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        checkpoint_fp="/wrk/mmajursk/small-data-cnns/UNet-Gan/ugan-hes-20190315T143809/checkpoint/model.ckpt"
        echo "Launching Training Script"
        python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" --use_augmentation=0 --restore_checkpoint_filepath="$checkpoint_fp" --restore_var_common_name=encoder | tee "$results_dir/log.txt"
    fi

    results_dir="$wrk_directory/$experiment_name-$N-$rep-gan-aug"
    if [ ! -d "$results_dir" ]; then
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        checkpoint_fp="/wrk/mmajursk/small-data-cnns/UNet-Gan/ugan-hes-20190315T143809/checkpoint/model.ckpt"
        echo "Launching Training Script"
        python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" --use_augmentation=1 --restore_checkpoint_filepath="$checkpoint_fp" --restore_var_common_name=encoder | tee "$results_dir/log.txt"
    fi
done


# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"
