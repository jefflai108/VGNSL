#!/bin/bash 

stage=$1

if [ $stage -eq 0 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert10-1000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 1 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert10-2000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 2 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert10-4000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 3 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert10-8000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 4 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert10-12000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 5 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert10-16000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

############################################################################################################################

if [ $stage -eq 10 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert2-1000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 11 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert2-2000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 12 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert2-4000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 13 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert2-8000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 14 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert2-12000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

if [ $stage -eq 15 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/dpdpseg-hubert2-16000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-dpdp-pred_attn_list-83k-5k-5k.npy
fi 

############################################################################################################################

if [ $stage -eq 20 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-1000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

if [ $stage -eq 21 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-2000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

if [ $stage -eq 22 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-4000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

if [ $stage -eq 23 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-8000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

if [ $stage -eq 24 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-12000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

if [ $stage -eq 25 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-16000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

if [ $stage -eq 26 ]; then 
    python src/simple_autowordse.py \
        --induced_tree_fpath /data/sls/scratch/clai24/data/SpokenCOCO/from_Freda/vgseg-vghubert-20000.pred.trees \
        --word_bounds_fpath /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
fi 

