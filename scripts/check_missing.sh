#!/bin/bash 

# summarize according to Karpathy split: 
# split: {image_key: [(wav_path, transcript_path, tree_path, alignment_path)]}

python data/check_missing.py \
    -s data/SpokenCOCO/karpathy_split.json -d data/SpokenCOCO -n wavs-speaker -o data/SpokenCOCO/SpokenCOCO_summary.json \
    -v data/SpokenCOCO/SpokenCOCO_val_wparse.json -t data/SpokenCOCO/SpokenCOCO_train_wparse.json -m no_alignment_wavs.txt
