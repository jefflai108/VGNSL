#!/bin/bash

# convert word-level tree to phone-level based on force alignment

data_dir=data/SpokenCOCO
data_split=val
vis_fpath=phn_tree_viz_for_Freda/${data_split}-phn_level_ground_truth.txt
python data/generate_phn-level_tree.py --data_split $data_split \
    --data_dir ${data_dir}/Freda-formatting \
    --data_summary_json ${data_dir}/SpokenCOCO_summary-83k-5k-5k.json > $vis_fpath
