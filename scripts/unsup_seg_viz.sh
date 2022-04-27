#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=hubert2; feature_dim=768
jason_feats=$4
discovery_type=attn # seg_feats + attention boundaries
last_ckpt=$5

# unsupervised segmentation based VGNSL 
for ckpt in `seq -s ' ' 0 ${last_ckpt}`; do
    datadir=data/SpokenCOCO
    expname=unsup_${discovery_type}_discovery_${jason_feats}_seg_feats_embed${embed_size}_MLPcombineV3_lr${lr}_${basename} # mlp_combine_v3 + deeper_score
    expdir=exp/spokencoco/${expname}
    echo $expdir
    vis_fpath=tree_viz_for_Freda/${expname}_ckpt${ckpt}.txt
    vis_fpath=tree_viz_for_Freda/${expname}_ckpt${ckpt}-val25000.txt # for Freda
    if [ -f ${expdir}/${ckpt}.pth.tar ]; then
        echo evaluating ${ckpt}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${ckpt}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
                           --visual_tree --visual_samples 2000 --data_split val > $vis_fpath
    fi 
done 
