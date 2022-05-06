#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=hubert2; feature_dim=768
jason_feats=mbr_104_1030_top10
jason_feats=$4
seg_feats_feature=disc-81_snapshot15_layer0
seg_feats_feature=$5
discovery_type=attn # seg_feats + attention boundaries

datadir=data/SpokenCOCO
expdir=exp/spokencoco/mbr_unsup_${discovery_type}_discovery_${jason_feats}_mbr_seg_feats_${seg_feats_feature}_embed${embed_size}_MLPcombineV2_lr${lr}_${basename} # mlp_combine_v2 + deeper_score
#expdir=exp/spokencoco/mbr_unsup_${discovery_type}_discovery_${jason_feats}_mbr_seg_feats_${seg_feats_feature}_embed${embed_size}_MLPcombineV3_lr${lr}_${basename} # mlp_combine_v3 + deeper_score
echo $expdir

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done
