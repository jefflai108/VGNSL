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
split=83k-5k-5k
data_split=$1
feature=$2
python data/check_longest_segment.py \
    -j ${datadir}/SpokenCOCO_summary-${split}.json -i ${datadir}/SpokenCOCO_images.h5 \
    -o ${datadir}/Freda-formatting/ -n 0 -l -1 -s $data_split -f logmelspec
