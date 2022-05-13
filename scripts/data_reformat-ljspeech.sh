#!/bin/bash

datadir=data/LJspeech
data_split=dev
data_split=$1
feature=hubert
feature=$2
layer_num=2
layer_num=$3

python data/data_reformat_v4-ljspeech.py \
    --data-dir $datadir --output-dir ${datadir}/Freda-formatting/ --h5_format \
    --data-split $data_split --feature $feature --layer_num $layer_num
