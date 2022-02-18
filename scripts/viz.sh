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
last_ckpt=$5

for ckpt in `seq -s ' ' 0 ${last_ckpt}`; do
    datadir=data/SpokenCOCO
    expname=force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
    expdir=exp/spokencoco/${expname}
    vis_fpath=tree_viz_for_Freda/${expname}_ckpt${ckpt}.txt
    if [ -f ${expdir}/${ckpt}.pth.tar ]; then
        echo evaluating ${ckpt}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${ckpt}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} --visual_tree > $vis_fpath
    fi 
done 
