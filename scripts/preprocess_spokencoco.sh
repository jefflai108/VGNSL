#!/bin/bash 

# ln -s original wavs/ to wavs-speaker (w/ speaker-split sub-directories), 
# then extract and store transcript & parse tree into separate .txt files in wavs-speaker 
# finally, create a vocab dictionary based on the transcripts
# ** remember to ln -s FULL PATH ** 

datadir=data/SpokenCOCO
python data/preprocess_spokencoco.py \
    -v ${datadir}/SpokenCOCO_val_wparse.json -t ${datadir}/SpokenCOCO_train_wparse.json \
    -d ${datadir} -n wavs-speaker -p ${datadir}/SpokenCOCO_vocab.pkl
