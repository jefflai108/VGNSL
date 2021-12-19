#!/bin/bash

# script for force alignment with Montreal Force Aligner 
# https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps
# 
# since MFA on the entire corpus takes too long, we do it in parallel by speaker-split 

start_speaker=$1
end_speaker=$2

conda activate aligner 

datadir=data/SpokenCOCO
targetdir=test_force_aligner
split=train

for speaker in $(seq $start_speaker $end_speaker); do 
    mfa align $datadir/wavs/$split/$speaker/ $datadir/librispeech-lexicon.txt english $datadir/wavs-aligned/$split/$speaker/ -t $datadir/.mfa_dump-${split}-${speaker} -j 5 -v --debug --clean
done
