#!/bin/bash

# convert word-level tree to phone-level based on force alignment

data_dir=data/SpokenCOCO
python data/generate_phn-level_tree.py \
    --data_dir ${data_dir}/Freda-formatting \
    --data_summary_json ${data_dir}/SpokenCOCO_summary-100-100-100.json
