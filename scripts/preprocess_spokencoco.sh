#!/bin/bash 

# ln -s original wavs/ to wavs-speaker (w/ speaker-split sub-directories), 
# then extract and store transcript & parse tree into separate .txt files in wavs-speaker 
# finally, create a vocab dictionary based on the transcripts
# ** remember to ln -s FULL PATH ** 

datadir=data/SpokenCOCO
python data/preprocess_spokencoco.py \
    -v ${datadir}/Freda-formatting/val_phn_caps-83k-5k-5k.txt \
    -t ${datadir}/Freda-formatting/train_phn_caps-83k-5k-5k.txt \
    -d ${datadir} -n wavs-speaker -p ${datadir}/SpokenCOCO_phn_vocab-threshold1.pkl
exit 0

python data/preprocess_spokencoco.py \
    -v ${datadir}/SpokenCOCO_val_wparse.json -t ${datadir}/SpokenCOCO_train_wparse.json \
    -d ${datadir} -n wavs-speaker -p ${datadir}/SpokenCOCO_vocab-threshold1.pkl
