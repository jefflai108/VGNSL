#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=logmelspec; feature_dim=40
feature=hubert; feature_dim=768
feature=hubert2; feature_dim=768
feature=hubert4; feature_dim=768
feature=hubert6; feature_dim=768
feature=hubert8; feature_dim=768
feature=hubert10; feature_dim=768
feature=$4
if [[ "$feature" = "logmelspec" ]]; then feature_dim=40 ; fi
margin=$5
head_init_bias=$6
rl_loss=$7

datadir=data/SpokenCOCO
if [[ $rl_loss ]]; then 
    vse_reward_alpha=$rl_loss 
    expdir=exp/spokencoco/phn_force_aligned_diffboundV2-gtword_whole_${feature}_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_vseRL${vse_reward_alpha}_${basename}
else
    vse_reward_alpha=1.0
    expdir=exp/spokencoco/phn_force_aligned_diffboundV2-gtword_whole_${feature}_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_${basename}
fi 

echo $expdir
word_seg_pth=${expdir}/predict_word_seg
mkdir -p $word_seg_pth

i=0
while [ $i -ne 30 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
                           --predict_word_seg_path ${word_seg_pth}/${i}_pred_word_seg.npy
    fi 
    i=$(($i+1))
done
