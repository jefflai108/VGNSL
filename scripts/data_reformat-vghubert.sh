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
split=$1
num_labs=$2
lab_id=$3
#for data_split in train val test; do
data_split=$4
layer_num=

python data/data_reformat_v4-vghubert.py \ 
    -j ${datadir}/SpokenCOCO_summary-${split}.json \
    -o ${datadir}/Freda-formatting/ --h5_format --parallelize -n $num_labs -l $lab_id \
    --data-split $data_split --feature hubert --layer_num $layer_num

