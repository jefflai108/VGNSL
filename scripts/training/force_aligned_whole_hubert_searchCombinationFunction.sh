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
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_MLPcombine_lr${lr}_${basename} # mlp_combine
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_MLPcombineV2_lr${lr}_${basename} # mlp_combine_v2 + deeper_score
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_MLPcombineV3_lr${lr}_${basename} # mlp_combine_v3 + deeper_score
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 5 --batch_size 128 --margin 0.2 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} \
    --speech_hdf5 --feature ${feature} --load_pretrained \
    --mlp_combine_v3 --deeper_score
    #--mlp_combine_v2 --deeper_score
    #--mlp_combine
