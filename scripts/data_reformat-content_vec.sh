#!/bin/bash

# reformat into Freda's data pre-processing format. 
# pre-extract and pre-store SpokenCOCO_summary.json: 
# for each split: 
#   image: in .npy file 
#   text/tree: directly in plain text file 
#   utterances: pre-extract aligned features (parallelizable) 
# 
# example usgaes:
# ./scripts/data_reformat.sh 83k-5k-5k 20 0 train 
# ./scripts/data_reformat.sh 83k-5k-5k 20 1 train
# ./scripts/data_reformat.sh 83k-5k-5k 20 2 train

datadir=data/SpokenCOCO
#for split in 83k-5k-5k 10k-5k-5k 10k-1k-1k; do 
split=$1
num_labs=$2
lab_id=$3
#for data_split in train val test; do
data_split=$4
feature=$5
layer_num=12
python data/data_reformat_v4.py \
    -j ${datadir}/SpokenCOCO_summary-${split}.json -i ${datadir}/SpokenCOCO_images.h5 \
    -o ${datadir}/Freda-formatting/ --h5_format --parallelize -n $num_labs -l $lab_id \
    --data-split $data_split --feature $feature --layer_num $layer_num
