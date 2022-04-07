#!/bin/bash 

# create word_list file with uniform word segmentations

feature=$1
datadir=data/SpokenCOCO/Freda-formatting
for data_split in test val train; do
    python data/create_uniform_word_seg.py \
        --orig_word_seg_file ${datadir}/${data_split}_segment-${feature}_word_list-83k-5k-5k.npy \
        --output_word_seg_file ${datadir}/${data_split}_segment-${feature}_uniform_word_list-83k-5k-5k.npy
done 
