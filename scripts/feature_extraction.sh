#!/bin/bash 

# pre-extract and store image embeddings into .hdf5 files 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/tools/Jacinle

python data/extract_coco_features.py \
    --val-caption data/SpokenCOCO/SpokenCOCO_val_wparse.json \
    --train-caption data/SpokenCOCO/SpokenCOCO_train_wparse.json \
    --image-root data/mscoco \
    --output data/SpokenCOCO/SpokenCOCO_images.h5 \
    --image-size 224 --batch-size 64 --use-gpu True 
