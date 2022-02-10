#!/bin/bash

basename=100-100-100
embed_size=512
lr=1e-4
feature=hubert; feature_dim=768
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_hubert_100-100-100
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 5 --batch_size 128 --margin 0.2 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} \
    --speech_hdf5 --feature ${feature} --load_pretrained 
