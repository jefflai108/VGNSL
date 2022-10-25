#!/bin/bash

# train kmeans on SpokenCOCO train set's speech features 
cluster=$1
hubert_layer=$2
python data/kmeans_on_dpdp.py --data_dir data/SpokenCOCO/Freda-formatting \
    --kmeans_dir data/SpokenCOCO/kmeans \
    --feature hubert${hubert_layer} --n_clusters $cluster --percent 1.0

## train kmeans on SpokenCOCO train set's speech features 
#cluster=$1
#python data/kmeans_on_mbr.py --data_dir data/SpokenCOCO/Freda-formatting \
#    --kmeans_dir data/SpokenCOCO/kmeans \
#    --feature hubert --n_clusters $cluster --percent 1.0

## train kmeans on SpokenCOCO train set's speech features 
#for cluster in 64 128 256 512 1024 2048; do
#    python data/kmeans.py --data_dir data/SpokenCOCO/Freda-formatting \
#        --kmeans_dir data/SpokenCOCO/kmeans \
#        --feature $1 --n_clusters $cluster --percent 0.2
#done 
