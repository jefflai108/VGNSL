#!/bin/bash

# script for force alignment with Montreal Force Aligner 
# https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps

conda activate aligner 

datadir=data/SpokenCOCO
targetdir=test_force_aligner
targetdir=wavs
mfa align $datadir/$targetdir/ $datadir/librispeech-lexicon.txt english $datadir/${targetdir}-aligned/ -t $datadir/.mfa_dump -j 1 -v --debug --clean
