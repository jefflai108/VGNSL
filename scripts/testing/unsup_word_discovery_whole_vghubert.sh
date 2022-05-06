#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=disc-82_0; feature_dim=768
feature=$4
jason_feats=disc-81_spokencoco_preFeats_max_0.7_9_clsAttn
jason_feats=$5
discovery_type=attn # change to word later

datadir=data/SpokenCOCO
expdir=exp/spokencoco/unsup_${discovery_type}_discovery_${jason_feats}_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
echo $expdir

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done