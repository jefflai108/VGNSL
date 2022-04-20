import collections
import itertools
import numpy as np
from tqdm import tqdm
import os
import os.path as osp

from evaluation import _retrieve_text_from_tree, _cleanup_tree, extract_statistics, extract_spans
from evaluation import f1_score as corpus_level_f1_score

def mbr_selection(samples, key_function):
    ''' 
    MBR selection function
    Input: 
        - samples: list of samples
        - key_function: takes two samples, and returns a number indicating agreement
    Output:
        - a dictionary containing the best sample and its agreement score
    '''
    sample_cnt = collections.Counter(samples)
    sorted_samples = sorted(sample_cnt.keys())
    sample_pairs = itertools.product(sorted_samples, sorted_samples)
    keys = np.array(
        list(map(lambda x: key_function(*x) * sample_cnt[x[1]], sample_pairs))  
        # weight second item contributes to the first
    ).reshape(len(sorted_samples), -1).sum(1)
    best_position = keys.argmax()

    return {
        'best_sample': sorted_samples[best_position],
        'best_key': keys[best_position]
    }
    
def pairwise_f1_score_for_mbr(orig_produced_trees, orig_gold_trees):
    # hack 
    orig_produced_trees = [orig_produced_trees]
    orig_gold_trees = [orig_gold_trees]

    # cleanup 
    orig_produced_trees, orig_gold_trees, _ = _cleanup_tree(orig_produced_trees, orig_gold_trees)

    # double-check underlying word/phn sequence match 
    orig_gold_trees_text = [_retrieve_text_from_tree(orig_gold_tree) for orig_gold_tree in orig_gold_trees]
    orig_produced_trees_text = [_retrieve_text_from_tree(orig_produced_tree) for orig_produced_tree in orig_produced_trees]
    assert orig_gold_trees_text == orig_produced_trees_text # underlying words/phones should match. 

    # compute f1 score over spans 
    gold_trees = list(map(lambda tree: extract_spans(tree), orig_gold_trees))
    produced_trees = list(map(lambda tree: extract_spans(tree), orig_produced_trees))
    assert len(produced_trees) == len(gold_trees), print(len(produced_trees), len(gold_trees))
    precision_cnt, precision_denom, recall_cnt, recall_denom = 0, 0, 0, 0
    for i, item in enumerate(produced_trees):
        pc, pd, rc, rd = extract_statistics(gold_trees[i], item)
        precision_cnt += pc
        precision_denom += pd
        recall_cnt += rc
        recall_denom += rd
    precision = float(precision_cnt) / precision_denom * 100.0
    recall = float(recall_cnt) / recall_denom * 100.0
    f1 = 2 * precision * recall / (precision + recall)
       
    return f1

def _open_tree_file(tree_file): 
    with open(tree_file, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    return content

def run(pred_tree_dir_list): 

    # open all ckpt's pred_trees
    pred_tree_files = []
    for pred_tree_dir in pred_tree_dir_list:
        pred_tree_files.extend([osp.join(pred_tree_dir, f) for f in os.listdir(pred_tree_dir) if osp.isfile(osp.join(pred_tree_dir, f))])
    pred_trees = [_open_tree_file(pred_tree_file) for pred_tree_file in pred_tree_files]

    # mbr selection across ckpts
    mbr_selected_trees = []
    for i in tqdm(range(25000)): 
        pred_tree_samples = [pred_tree[i] for pred_tree in pred_trees] # outputs from different model/ckpt given an input
        output = mbr_selection(pred_tree_samples, key_function=pairwise_f1_score_for_mbr)
        mbr_selected_trees.append(output['best_sample'])

    # calculate corpus-level F1 against ground-truth 
    ground_truth_trees = _open_tree_file('data/SpokenCOCO/Freda-formatting/test_word-level-ground-truth-83k-5k-5k.txt')
    f1, _, _ = corpus_level_f1_score(mbr_selected_trees, ground_truth_trees)
    print(f'MBR F1 for {pred_tree_dir} is {f1:.3f}')

if __name__ == '__main__':

    ## MBR selection for MFA whole_hubert
        
    # MBR across epoches and learning rates but the same hubert layers
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr'])
    exit()

    # MBR across epoches and hubert layers but the same learning rate
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr']) # 55.561

    # MBR across epoches but within a hyper-parameter set 
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr']) # 54.994
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr']) # 53.246
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr']) # 53.460
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr']) # 53.143
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr']) # 36.669
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr']) # 48.505

