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

# reduce num_epochs from 20 --> 10 to speed up model dev cycle 
# reduce batch_size from 128 --> 64 to avoid mem error on SLS machines 
# increase val_step from 500 --> 1500 to speed up model dev cycle 
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 30 --workers 0 --batch_size 128 --margin ${margin} --val_step 1000 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} --lambda_hi ${head_init_bias} --vse_reward_alpha $vse_reward_alpha \
    --speech_hdf5 --feature ${feature} --load_pretrained --phn_force_align --diffbound_gtword \
    --init_with_uniform_word_force_align $uniform_vgnsl
