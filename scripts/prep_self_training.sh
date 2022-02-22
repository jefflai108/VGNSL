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
best_ckpt=$5
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_${basename}

# Step 1: export the given pretrained model ckpt's inferred tree to file 
if [ -f ${expdir}/${best_ckpt}.pth.tar ]; then
echo evaluating ${best_ckpt}.pth.tar
for data_split in val train; do 
    python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${best_ckpt}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
                       --export_tree --data_split ${data_split} --export_tree_path ${expdir}/${data_split}_ground-truth-${basename}.txt
done 
fi 
