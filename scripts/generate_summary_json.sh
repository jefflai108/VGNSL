#!/bin/bash 

# summarize according to Karpathy split: 
# split: {image_key: [(wav_path, transcript_path, tree_path, alignment_path)]}
#
# There are 617222 total utterances. 5 has no alignments. 384 exceeds 15 seconds

datadir=data/SpokenCOCO
python data/generate_summary_json.py \
    -s ${datadir}/karpathy_split.json -d ${datadir} -n wavs-speaker -o ${datadir}/SpokenCOCO_summary-100-100-100.json \
    -v ${datadir}/SpokenCOCO_val_wparse.json -t ${datadir}/SpokenCOCO_train_wparse.json
