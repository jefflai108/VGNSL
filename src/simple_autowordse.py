import sys 
import os
from tqdm import tqdm 
import argparse

import numpy as np

sys.path.insert(-1, os.path.join(sys.path[0], '/data/sls/scratch/clai24/syntax/AutowordSE/src'))
from autowordse import BareTree, ExtendedTree, TreeAligner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--induced_tree_fpath', type=str)
    parser.add_argument('--word_bounds_fpath', type=str)
    args = parser.parse_args()
    
    datum_files = {
        'gold_tree_fpath': '/data/sls/scratch/clai24/syntax/VGNSL/data/SpokenCOCO/Freda-formatting/test_word-level-ground-truth-83k-5k-5k.txt',
        'induced_tree_fpath': '/data/sls/scratch/clai24/syntax/VGNSL/exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer9_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr/1_pred_tree.txt',
        'oracle_bounds_fpath': '/data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test_segment-hubert2_word_list-83k-5k-5k.npy',
        'word_bounds_fpath': '/data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_attn_list-83k-5k-5k.npy' # /data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test-mbr_104_1030_top10-pred_word_list-83k-5k-5k.npy
    }

    datum_files = {
        'gold_tree_fpath': '/data/sls/scratch/clai24/syntax/VGNSL/data/SpokenCOCO/Freda-formatting/test_word-level-ground-truth-83k-5k-5k.txt',
        'induced_tree_fpath': args.induced_tree_fpath, 
        'oracle_bounds_fpath': '/data/sls/scratch/clai24/data/SpokenCOCO/Freda-formatting/test_segment-hubert2_word_list-83k-5k-5k.npy',
        'word_bounds_fpath': args.word_bounds_fpath
    }
    
    with open(datum_files['gold_tree_fpath'], 'r') as f: 
        gold_trees = f.readlines()
    gold_trees = [x.strip('\n') for x in gold_trees]
    
    with open(datum_files['induced_tree_fpath'], 'r') as f: 
        induced_trees = f.readlines()
    induced_trees = [x.strip('\n') for x in induced_trees]

    oracle_bounds = np.load(datum_files['oracle_bounds_fpath'], allow_pickle=True)[0]
    for i, oracle_bound in oracle_bounds.items(): 
        oracle_bound_in_time = [(word, start_time * 0.02, end_time * 0.02) for (word, start_time, end_time) in oracle_bound]
        oracle_bounds[i] = oracle_bound_in_time

    word_bounds = np.load(datum_files['word_bounds_fpath'], allow_pickle=True)[0]

    sum_predicted_avg_iou = 0 
    sum_rbt_avg_iou = 0 
    real_cnt = 0
    for idx in tqdm(range(0, 25000)):

        if gold_trees[idx] == 'MISMATCH': 
            continue 

        real_cnt += 1
        datum = {
            'gold_tree': BareTree.fromstring(gold_trees[idx]),
            'induced_tree': BareTree.fromstring(induced_trees[idx]),
            'oracle_bounds': oracle_bounds[idx],
            'word_bounds': word_bounds[idx]
        }

        if len(word_bounds[idx]) >= 2:
            predicted_tree = ExtendedTree.from_tree_and_leaf_timestamps(datum['induced_tree'], datum['word_bounds'])
            gold_tree = ExtendedTree.from_tree_and_leaf_timestamps(datum['gold_tree'], datum['oracle_bounds'])
            n_leaves = len(datum['induced_tree'].leaves())        
            bare_rbt = f'( {n_leaves-2} {n_leaves-1} )'
            for j in range(n_leaves-3, -1, -1):
                bare_rbt = f'( {j} {bare_rbt} )'
            bare_rbt = BareTree.fromstring(bare_rbt)
            right_branching_tree = ExtendedTree.from_tree_and_leaf_timestamps(bare_rbt, datum['word_bounds'])
            
            predicted_tree_aligner = TreeAligner(predicted_tree, gold_tree)
            predicted_score, predicted_alignment = predicted_tree_aligner()
            predicted_avg_iou = predicted_score * 2 / (predicted_tree.n_nodes() + gold_tree.n_nodes())

            right_branching_tree_aligner = TreeAligner(right_branching_tree, gold_tree)
            rbt_score, rbt_alignment = right_branching_tree_aligner()
            rbt_avg_iou = rbt_score * 2 / (predicted_tree.n_nodes() + gold_tree.n_nodes())  # n_node of rbt is n_node of predicted

        else: # give 0% to IOUs
            #print(idx)
            #print(gold_trees[idx])
            #print(induced_trees[idx])
            #print(oracle_bounds[idx])
            #print(word_bounds[idx])
            predicted_avg_iou = 0 
            rbt_avg_iou = 0

        sum_predicted_avg_iou += predicted_avg_iou
        sum_rbt_avg_iou += rbt_avg_iou

    print(sum_predicted_avg_iou / real_cnt, sum_rbt_avg_iou / real_cnt)
