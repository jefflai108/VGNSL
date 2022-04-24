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

if [[ "$feature" = "hubert2" ]]; then uniform_vgnsl=exp/spokencoco/uniform_force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/1.pth.tar; fi
if [[ "$feature" = "hubert4" ]]; then uniform_vgnsl=exp/spokencoco/uniform_force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/1.pth.tar; fi
if [[ "$feature" = "hubert6" ]]; then uniform_vgnsl=exp/spokencoco/uniform_force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/1.pth.tar; fi
if [[ "$feature" = "hubert8" ]]; then uniform_vgnsl=exp/spokencoco/uniform_force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/1.pth.tar; fi
if [[ "$feature" = "hubert10" ]]; then uniform_vgnsl=exp/spokencoco/uniform_force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/0.pth.tar; fi
if [[ "$feature" = "hubert" ]]; then uniform_vgnsl=exp/spokencoco/uniform_force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/1.pth.tar; fi

datadir=data/SpokenCOCO
if [[ $rl_loss ]]; then 
    vse_reward_alpha=$rl_loss 
    expdir=exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_${feature}_uniform-seg-init_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_vseRL${vse_reward_alpha}_${basename}
else
    vse_reward_alpha=1.0
    expdir=exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_${feature}_uniform-seg-init_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_${basename}
fi 

i=0
while [ $i -ne 30 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done
