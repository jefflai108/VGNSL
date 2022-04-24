#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=hubert_large16; feature_dim=1024
feature=hubert_large18; feature_dim=1024
feature=hubert_large24; feature_dim=1024
feature=$4
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_${feature}_attnnorm_embed${embed_size}_lr${lr}_${basename}
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
echo $expdir
mkdir -p $expdir/mbr
i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
                           --mbr_path ${expdir}/mbr/${i}_pred_tree.txt
    fi 
    i=$(($i+1))
done
exit 0
