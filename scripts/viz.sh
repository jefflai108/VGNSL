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
ckpt=$5
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
if [ -f ${expdir}/${ckpt}.pth.tar ]; then
    echo evaluating ${ckpt}.pth.tar
    python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${ckpt}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} --visual_tree
fi 
