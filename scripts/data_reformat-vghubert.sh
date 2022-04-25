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
split=83k-5k-5k
num_labs=0
lab_id=-1
#for data_split in train val test; do
data_split=$1
feature=$2
layer_num=$3

python data/data_reformat_v4-vghubert.py \
    -j ${datadir}/SpokenCOCO_summary-${split}.json \
    -o ${datadir}/Freda-formatting/ --h5_format -n $num_labs -l $lab_id \
    --data-split $data_split --feature $feature --layer_num $layer_num
