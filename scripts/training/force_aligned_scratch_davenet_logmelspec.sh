#!/bin/bash

basename=$1
embed_size=$2
lr=$3
davenet_embed_type=$4
feature=logmelspec; feature_dim=40
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_scratch-${davenet_embed_type}_${feature}_embed${embed_size}_lr${lr}_${basename}
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 5 --batch_size 32 --margin 0.2 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} \
    --speech_hdf5 --feature ${feature} --load_pretrained \
    --davenet_embed --davenet_embed_type $davenet_embed_type
