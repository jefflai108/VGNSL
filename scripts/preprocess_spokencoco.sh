#!/bin/bash 

# ln -s original wavs/ to wavs-speaker (w/ speaker-split sub-directories), 
# then extract and store transcript & parse tree into separate .txt files in wavs-speaker 

python data/preprocess_spokencoco.py \
    -v data/SpokenCOCO/SpokenCOCO_val_wparse.json -t data/SpokenCOCO/SpokenCOCO_train_wparse.json -d data/SpokenCOCO -n wavs-speaker
