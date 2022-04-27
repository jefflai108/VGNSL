#!/bin/bash

basename=$1
embed_size=$2
lr=$3
feature=hubert2; feature_dim=768
feature=$4
dino_feature=$5
if [[ "$dino_feature" = "vits8" ]]; then dino_dim=384; fi
if [[ "$dino_feature" = "vits16" ]]; then dino_dim=384; fi
if [[ "$dino_feature" = "vitb8" ]]; then dino_dim=768; fi
if [[ "$dino_feature" = "vitb16" ]]; then dino_dim=768; fi
if [[ "$dino_feature" = "deit_base_patch16_224" ]]; then dino_dim=1000; fi
if [[ "$dino_feature" = "deit_base_distilled_patch16_384" ]]; then dino_dim=1000; fi

datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_freezed_${dino_feature}_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim $dino_dim --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 2 --batch_size 128 --margin 0.2 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} \
    --speech_hdf5 --feature ${feature} --load_pretrained --dino_feature $dino_feature
