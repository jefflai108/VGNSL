#!/bin/bash 

# pre-extract and store image embeddings into .hdf5 files 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/tools/Jacinle

# deit-based image embed
python data/extract_dino_coco_features.py \
    --val-caption data/SpokenCOCO/SpokenCOCO_val_wparse.json \
    --train-caption data/SpokenCOCO/SpokenCOCO_train_wparse.json \
    --image-root data/mscoco \
    --output data/SpokenCOCO/SpokenCOCO_deit_base_distilled_patch16_384_images.h5 \
    --image-size 384 --batch-size 64 --use-gpu True 

# dino-based image embed
python data/extract_dino_coco_features.py \
    --val-caption data/SpokenCOCO/SpokenCOCO_val_wparse.json \
    --train-caption data/SpokenCOCO/SpokenCOCO_train_wparse.json \
    --image-root data/mscoco \
    --output data/SpokenCOCO/SpokenCOCO_dino_vits8_images.h5 \
    --image-size 224 --batch-size 64 --use-gpu True 

# resnet-based image embed 
python data/extract_coco_features.py \
    --val-caption data/SpokenCOCO/SpokenCOCO_val_wparse.json \
    --train-caption data/SpokenCOCO/SpokenCOCO_train_wparse.json \
    --image-root data/mscoco \
    --output data/SpokenCOCO/SpokenCOCO_images.h5 \
    --image-size 224 --batch-size 64 --use-gpu True 
