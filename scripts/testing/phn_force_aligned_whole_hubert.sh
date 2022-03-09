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
margin=$5
head_init_bias=$6
rl_loss=$7

datadir=data/SpokenCOCO
if [[ $rl_loss ]]; then 
    vse_reward_alpha=$rl_loss 
    expdir=exp/spokencoco/phn_force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_vseRL${vse_reward_alpha}_${basename}
else
    vse_reward_alpha=1.0
    expdir=exp/spokencoco/phn_force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_${basename}
fi 
i=0
while [ $i -ne 10 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_phn_vocab-threshold1.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done
exit 0
