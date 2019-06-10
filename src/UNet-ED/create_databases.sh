#!/bin/bash

python build_lmdb_ed.py --image_folder=/scratch/small-data-cnns/source_data/HES_raw --output_filepath=/scratch/small-data-cnns/source_data/HES --dataset_name=hes

python build_lmdb_ed.py --image_folder=/scratch/small-data-cnns/source_data/RPE_raw --output_filepath=/scratch/small-data-cnns/source_data/RPE --dataset_name=rpe
