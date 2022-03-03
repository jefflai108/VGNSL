#!/bin/bash

# (1) convert word-level tree to phone-level based on force alignment
# example commands: 
# ./scripts/generate_phn-level_tree.sh test 
# ./scripts/generate_phn-level_tree.sh val
# 
# (2) convert word_list to phn_list for logmelspec/hubert based on force alignments
# example commands: 
# ./scripts/generate_phn-level_tree.sh train logmelspec 0
# ./scripts/generate_phn-level_tree.sh test logmelspec 0
# ./scripts/generate_phn-level_tree.sh hubert 12 

data_dir=data/SpokenCOCO
data_split=$1
vis_fpath=phn_tree_viz_for_Freda/${data_split}-phn_level_ground_truth.txt
python data/generate_phn-level_tree.py --data_split $data_split \
    --data_dir ${data_dir}/Freda-formatting \
    --data_summary_json ${data_dir}/SpokenCOCO_summary-83k-5k-5k.json \
    --feature $2 --layer_num $3
    #> $vis_fpath
