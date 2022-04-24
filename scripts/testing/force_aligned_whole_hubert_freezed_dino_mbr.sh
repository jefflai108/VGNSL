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
dino_feature=$5
if [[ "$dino_feature" = "vits8" ]]; then dino_dim=384; fi
if [[ "$dino_feature" = "vits16" ]]; then dino_dim=384; fi
if [[ "$dino_feature" = "vitb8" ]]; then dino_dim=768; fi
if [[ "$dino_feature" = "vitb16" ]]; then dino_dim=768; fi
if [[ "$dino_feature" = "deit_base_patch16_224" ]]; then dino_dim=1000; fi
if [[ "$dino_feature" = "deit_base_distilled_patch16_384" ]]; then dino_dim=1000; fi

datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_freezed_${dino_feature}_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
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
exit 0
