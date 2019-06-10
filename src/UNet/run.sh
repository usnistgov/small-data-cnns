#!/bin/bash

# for i in `seq 1 8`;
# do
#     sbatch launch_train_sbatch_rpe.sh
#     sleep 10
# done


for i in `seq 1 4`;
do
    sbatch launch_train_sbatch_rpe2.sh
    sleep 10
done