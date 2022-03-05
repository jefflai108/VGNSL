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
feature=content_vec_v07_1112; feature_dim=768
feature=content_vec_v12_0512; feature_dim=768
feature=$4
margin=$5
head_init_bias=$6
datadir=data/SpokenCOCO
expdir=exp/spokencoco/phn_force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_margin${margin}_lambdahi${head_init_bias}_${basename}
# reduce num_epochs from 20 --> 10 to speed up model dev cycle 
# reduce batch_size from 128 --> 64 to avoid mem error on SLS machines 
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_phn_vocab-threshold1.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 10 --workers 0 --batch_size 64 --margin ${margin} --val_step 1500 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} --lambda_hi ${head_init_bias} \
    --speech_hdf5 --feature ${feature} --load_pretrained --phn_force_align
