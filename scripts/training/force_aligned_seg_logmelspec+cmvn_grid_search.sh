#!/bin/bash

basename=$1
embed_size=$2
lr=$3
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_seg_logmelspec+cmvn_embed${embed_size}_lr${lr}_${basename}
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 10 --batch_size 256 --margin 0.2 \
    --embed_size ${embed_size} --logmelspec_dim 40 --learning_rate ${lr} --logmelspec_cmvn

