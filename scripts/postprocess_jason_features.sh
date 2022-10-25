#!/bin/bash 

datadir=data/SpokenCOCO

jason_feature=disc-26_spokencoco_preFeats_weightedmean_0.8_7_clsAttn
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn
jason_feature=model3_spokencoco_preFeats_weightedmean_0.8_7_clsAttn
jason_feature=disc-81_spokencoco_preFeats_max_0.7_9_clsAttn
jason_feature=disc-62_spokencoco_preFeats_mean_0.9_5_clsAttn
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.05_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.05_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.05_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.1_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.1_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.1_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.5_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.5_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.5_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.3_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.3_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.3_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.4_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.4_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.4_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadno_insertThreshold0.2_snapshotbest
jason_feature=disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadno_insertThreshold0.2_snapshotbest
jason_feature=disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadno_insertThreshold0.2_snapshotbest
jason_feature=mbr_104_1030_top10
jason_feature=dpdp

jason_feature=dpdp
split=$1
layer_num=$2
#for split in val test train; do
    python data/postprocess_jason_features_dpdp.py \
        -j ${datadir}/SpokenCOCO_summary-83k-5k-5k.json --data_dir ${datadir} \
        -o ${datadir}/Freda-formatting \
        --data_split $split --new_feature $jason_feature --vghubert_layer $layer_num
#done 
