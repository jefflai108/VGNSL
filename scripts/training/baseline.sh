#!/bin/bash

#cd src
datadir=data/SpokenCOCO
expdir=exp/spokencoco/text_spokencoco
python src/train.py --logger_name $expdir \
    --data_path ${datadir} --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --data_summary_json ${datadir}/SpokenCOCO_summary.json --image_hdf5 ${datadir}/SpokenCOCO_images.h5 \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 5 --batch_size 256 --margin 0.2 
