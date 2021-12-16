#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/tools/Jacinle

python data/extract_coco_features.py \
    --caption data/SpokenCOCO/SpokenCOCO_val_wparse.json \
    --image-root data/mscoco \
    --output data/SpokenCOCO/SpokenCOCO_val_images.h5 \
    --image-size 224 --batch-size 64 --use-gpu True 

python data/extract_coco_features.py \
    --caption data/SpokenCOCO/SpokenCOCO_train_wparse.json \
    --image-root data/mscoco \
    --output data/SpokenCOCO/SpokenCOCO_train_images.h5 \
    --image-size 224 --batch-size 64 --use-gpu True 

