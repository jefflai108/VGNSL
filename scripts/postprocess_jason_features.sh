#!/bin/bash 

datadir=data/SpokenCOCO
python data/postprocess_jason_features.py \
    -j ${datadir}/SpokenCOCO_summary-83k-5k-5k.json --data_dir ${datadir} \
    -o ${datadir}/Freda-formatting \
    --data_split $1 --new_feature disc-26_spokencoco_preFeats_weightedmean_0.8_7_clsAttn
