#!/bin/bash 

datadir=data/SpokenCOCO
jason_feature=disc-26_spokencoco_preFeats_weightedmean_0.8_7_clsAttn
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn
jason_feature=model3_spokencoco_preFeats_weightedmean_0.8_7_clsAttn
jason_feature=disc-81_spokencoco_preFeats_max_0.7_9_clsAttn
jason_feature=$1
for split in val test train; do
    python data/postprocess_jason_features.py \
        -j ${datadir}/SpokenCOCO_summary-83k-5k-5k.json --data_dir ${datadir} \
        -o ${datadir}/Freda-formatting \
        --data_split $split --new_feature $jason_feature
done 
