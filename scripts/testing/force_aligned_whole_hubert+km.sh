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
km_cluster=$5
stage=1

if [ $stage -eq 1 ]; then 
expdir=exp/spokencoco/force_aligned_whole_${feature}_km${km_cluster}-discrete_word-embed${embed_size}_lr${lr}_${basename}
fi 
if [ $stage -eq 2 ]; then 
expdir=exp/spokencoco/whole_${feature}_km${km_cluster}-discrete_phone-embed${embed_size}_lr${lr}_${basename}
fi 

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done
