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
feature=$4

datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_MLPcombine_lr${lr}_${basename} # mlp_combine
#expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_MLPcombineV2_lr${lr}_${basename} # mlp_combine_v2 + deeper_score
#expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_MLPcombineV3_lr${lr}_${basename} # mlp_combine_v3 + deeper_score

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done
exit 0
