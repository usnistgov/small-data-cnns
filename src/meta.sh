#!/usr/bin/env bash


# Concrete: 
# UNet no aug
# UNet with aug
# SegNet no aug
# SegNet with aug


# python build_lmdb.py --image_folder="/scratch/small-data-cnns/source_data/Concrete_orig/images" --mask_folder="/scratch/small-data-cnns/source_data/Concrete_orig/masks" --output_filepath="/scratch/small-data-cnns/source_data/" --dataset_name="Concrete" --train_fraction=0.8


cd UNet 
python /wrk/mmajursk/small-data-cnns/src/UNet/train_unet.py --batch_size=4 --train_database=/home/mmajursk/Gitlab/Semantic-Segmentation/data/train-Concrete.lmdb --test_database=/home/mmajursk/Gitlab/Semantic-Segmentation/data/test-Concrete.lmdb --output_dir=/home/mmajursk/Gitlab/Semantic-Segmentation/UNet/model --balance_classes=1
