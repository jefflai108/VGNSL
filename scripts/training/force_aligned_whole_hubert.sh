#!/bin/bash

basename=$1
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_hubert_${basename}
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 10 --batch_size 128 --margin 0.2 \
    --embed_size 512 --feature_dim 768 --speech_hdf5 --feature hubert

