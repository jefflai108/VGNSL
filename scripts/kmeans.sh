#!/bin/bash

# train kmeans on SpokenCOCO train set's speech features 

#for cluster in 64 128 256 512 1024; do
for cluster in 64 128; do
    python data/kmeans.py --data_dir data/SpokenCOCO/Freda-formatting \
        --kmeans_dir data/SpokenCOCO/kmeans \
        --feature $1 --n_clusters $cluster --percent 0.2
done 
