#!/bin/bash 

# summarize according to Karpathy split: 
# split: {image_key: [(wav_path, transcript_path, tree_path, alignment_path)]}
#
# There are 617222 total utterances. 5 has no alignments. 384 exceeds 15 seconds

python data/generate_summary_json.py \
    -s data/SpokenCOCO/karpathy_split.json -d data/SpokenCOCO -n wavs-speaker -o data/SpokenCOCO/SpokenCOCO_summary.json \
    -v data/SpokenCOCO/SpokenCOCO_val_wparse.json -t data/SpokenCOCO/SpokenCOCO_train_wparse.json
