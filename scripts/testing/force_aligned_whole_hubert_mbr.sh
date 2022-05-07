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
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
mkdir -p ${expdir}/mbr # for test
mkdir -p ${expdir}/mbr-self_train # for train/val 

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        for split in val train; do
            python src/test.py --data_split $split \
                               --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
                               --mbr_path ${expdir}/mbr-self_train/${i}_pred_tree-${split}.txt
        done
    fi 
    i=$(($i+1))
done
