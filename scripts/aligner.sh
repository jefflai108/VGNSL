#!/bin/bash

# script for force alignment with Montreal Force Aligner 
# https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps
# 
# for simply appling pre-trained Librispeech ASR, do `mfa align`; 
# for domain-adapt pre-trained Librispeech ASR (shifting GMM mean) and then align, do `mfa adapt` 
# 
# since MFA on the entire corpus takes too long, we do it in parallel (20 jobs) by speaker-split 
# **make sure that the sub-directories are organized by speakers**
# example running commmand for job ID 3 (remember to run ID 0 - ID 20, 21 in total):  
# . /data/sls/r/u/yungsung/home/miniconda3/bin/activate && ./scripts/aligner.sh 3
# 
# fault-tolerent: 
# for i in $(seq 1 100); do ./scripts/aligner.sh 3; done

#conda activate aligner 
# use Yung-Sung's latest MFA install
source /data/sls/r/u/yungsung/home/miniconda3/bin/activate

python data/parallel_aligner.py \
    -d data/SpokenCOCO/wavs-speaker -c data/SpokenCOCO -l done_alignment_speakers_list.txt -N 20 -n $1

#datadir=data/SpokenCOCO
#for spkdir in $datadir/wavs-speaker/*; do
#    #echo $spkdir
#    targetdir="${spkdir/wavs-speaker/wavs-speaker-aligned}"
#    #echo $targetdir
#
#    echo aligning $spkdir 
#    echo '$spkdir' >> done_alignment_speakers_list.txt
#    #mfa align $spkdir $datadir/librispeech-lexicon.txt english $targetdir -t $datadir/.mfa_dump-${spkdir} -j 10 -v --debug --clean
#
#    # adapt pre-trained Librispeech to SpokenCOCO, then align. API: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/adapt_acoustic_model.html
#    mfa adapt $spkdir $datadir/librispeech-lexicon.txt english $targetdir -t $datadir/.mfa_adapt-${spkdir} -j 10 -v --debug --clean
#done
