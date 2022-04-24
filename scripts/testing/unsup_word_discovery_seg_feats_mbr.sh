#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=hubert; feature_dim=768
feature=hubert2; feature_dim=768
feature=hubert4; feature_dim=768
feature=hubert6; feature_dim=768
feature=hubert8; feature_dim=768
feature=hubert10; feature_dim=768
jason_feats=disc-81_spokencoco_preFeats_max_0.7_9_clsAttn
jason_feats=$4
discovery_type=attn # seg_feats + attention boundaries

datadir=data/SpokenCOCO
expdir=exp/spokencoco/unsup_${discovery_type}_discovery_${jason_feats}_seg_feats_embed${embed_size}_lr${lr}_${basename}
echo $expdir 
mkdir -p ${expdir}/mbr 

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
                           --mbr_path ${expdir}/mbr/${i}_pred_tree.txt
    fi 
    i=$(($i+1))
done
