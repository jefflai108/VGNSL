#!/bin/bash

basename=$1
embed_size=$2
lr=$3
feature=hubert2; feature_dim=768
jason_feats=disc-81_spokencoco_preFeats_max_0.7_9_clsAttn
jason_feats=$4
discovery_type=attn # seg_feats + attention boundaries
datadir=data/SpokenCOCO
expdir=exp/spokencoco/unsup_${discovery_type}_discovery_${jason_feats}_seg_feats_embed${embed_size}_MLPcombine_lr${lr}_${basename} # mlp_combine
expdir=exp/spokencoco/unsup_${discovery_type}_discovery_${jason_feats}_seg_feats_embed${embed_size}_MLPcombineV2_lr${lr}_${basename} # mlp_combine_v2 + deeper_score
expdir=exp/spokencoco/unsup_${discovery_type}_discovery_${jason_feats}_seg_feats_embed${embed_size}_MLPcombineV3_lr${lr}_${basename} # mlp_combine_v3 + deeper_score
expdir=exp/spokencoco/unsup_${discovery_type}_discovery_${jason_feats}_seg_feats_embed${embed_size}_MLPcombineV4_lr${lr}_${basename} # mlp_combine_v4 + deeper_score
echo $expdir
python src/train.py --logger_name $expdir \
    --data_path ${datadir}/Freda-formatting --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} \
    --init_embeddings 0 --img_dim 2048 --scoring_hidden_dim 128 \
    --num_epochs 20 --workers 2 --batch_size 128 --margin 0.2 \
    --embed_size ${embed_size} --feature_dim ${feature_dim} --learning_rate ${lr} \
    --speech_hdf5 --feature ${feature} --load_pretrained \
    --unsup_word_discovery_feats $jason_feats --unsup_word_discovery_feat_type ${discovery_type} --use_seg_feats_for_unsup_word_discovery \
    --mlp_combine_v4 --deeper_score
    #--mlp_combine_v3 --deeper_score
    #--mlp_combine_v2 --deeper_score
    #--mlp_combine