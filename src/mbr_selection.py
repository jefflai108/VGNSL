import collections
import itertools
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import sys

from evaluation import _retrieve_text_from_tree, _cleanup_tree, extract_statistics, extract_spans
from evaluation import f1_score as corpus_level_f1_score

sys.path.insert(-1, os.path.join(sys.path[0], '../analysis'))
from ex_sparseval import corpus_f1 as ex_sparseval_f1

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
    orig_produced_trees, orig_gold_trees, _, _, _, _, _ = _cleanup_tree(orig_produced_trees, orig_gold_trees)

    # double-check underlying word/phn sequence match 
    orig_gold_trees_text = [_retrieve_text_from_tree(orig_gold_tree) for orig_gold_tree in orig_gold_trees]
    orig_produced_trees_text = [_retrieve_text_from_tree(orig_produced_tree) for orig_produced_tree in orig_produced_trees]
    #assert orig_gold_trees_text == orig_produced_trees_text # underlying words/phones should match. 

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
    if precision_denom == 0 and recall_denom == 0: 
        return 0
    precision = float(precision_cnt) / precision_denom * 100.0
    recall = float(recall_cnt) / recall_denom * 100.0
    if precision == 0 and recall == 0: 
        return 0
    f1 = 2 * precision * recall / (precision + recall)
       
    return f1

def _open_tree_file(tree_file): 
    with open(tree_file, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    return content

def run(pred_tree_dir_list, unsup_word_discovery_feats=None, unsup_word_discovery_feat_type=None, data_split='test', output_fpath=None): 
    # standard files
    ground_truth_trees = _open_tree_file(f'data/SpokenCOCO/Freda-formatting/{data_split}_word-level-ground-truth-83k-5k-5k.txt')
    if unsup_word_discovery_feats: 
        unsup_discovered_word_alignments = np.load(f'data/SpokenCOCO/Freda-formatting/{data_split}-{unsup_word_discovery_feats}-{unsup_word_discovery_feat_type}_alignment_via_max_weight_matching-83k-5k-5k.npy', allow_pickle=True)[0]
        unsup_discovered_word_alignments = list(unsup_discovered_word_alignments.values())

    # open all ckpt's pred_trees
    pred_tree_files = []
    for pred_tree_dir in pred_tree_dir_list:
        pred_tree_files.extend([osp.join(pred_tree_dir, f) for f in os.listdir(pred_tree_dir) if osp.isfile(osp.join(pred_tree_dir, f))])
    if data_split in ['val', 'train']: # filter out val/train tree files 
        pred_tree_files = [pred_tree_file for pred_tree_file in pred_tree_files if ('-' + data_split + '.txt') in pred_tree_file]
    pred_trees = [_open_tree_file(pred_tree_file) for pred_tree_file in pred_tree_files]

    # mbr selection across ckpts
    mbr_selected_trees = []
    for i in tqdm(range(len(ground_truth_trees))): 
        pred_tree_samples = [pred_tree[i] for pred_tree in pred_trees] # outputs from different model/ckpt given an input
        output = mbr_selection(pred_tree_samples, key_function=pairwise_f1_score_for_mbr)
        mbr_selected_trees.append(output['best_sample'])

    # calculate corpus-level F1 against ground-truth 
    if unsup_word_discovery_feats: 
        mbr_selected_trees, ground_truth_trees, unsup_discovered_word_alignments, _, _, _, _ = _cleanup_tree(mbr_selected_trees, ground_truth_trees, unsup_discovered_word_alignments)
        f1 = ex_sparseval_f1(ground_truth_trees, mbr_selected_trees, unsup_discovered_word_alignments, is_baretree=True) # careful of the ordering: gold_trees --> pred_trees
    else:
        f1, _, _ = corpus_level_f1_score(mbr_selected_trees, ground_truth_trees, data_split=data_split)
    print(f'MBR F1 for {pred_tree_dir} is {f1:.3f}')

    # write to output fpath 
    if output_fpath:
        with open(output_fpath, 'w') as f: 
            for mbr_tree in mbr_selected_trees: 
                f.write('%s\n' % mbr_tree)

if __name__ == '__main__':
    ########################################################## MBR selection for MBR_unsup-discovery MBR_seg_feats (fully-unsup setting) ##################################################################
    # MBR across epoches and hubert layers but the same learning rate
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer9_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer10_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer11_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'],
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')

    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer0_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer1_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer2_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer3_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer4_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer5_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer6_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer7_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer8_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer9_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer10_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer11_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'],
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')

    # MBR across epoches but within a hyper-parameter set
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer0_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer1_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer2_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer3_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer4_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer5_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer6_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer7_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer8_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer9_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer10_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer11_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'mbr_104_1030_top10', 
        unsup_word_discovery_feat_type = 'attn')

    ######################################################################## write MBR selection outputs to file ###########################################################################################
    # write MBR decode outputs to file for phn_force_aligned_diffboundV1-gtword_whole_huberts_embed512_lr5e-6_83k-5k-5k
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_huberts_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_huberts_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr', 
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_huberts_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_83k-5k-5k.txt')

    # write MBR decode outputs to file for force_aligned_freezed_{vits16,vits8,vitb8,vitb16,deits}_whole_hubert2_embed512_lr5e-6_83k-5k-5k
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
         data_split = 'val', 
         output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_huberts_and_freezed_vits_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
         data_split = 'train', 
         output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_huberts_and_freezed_vits_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
         data_split = 'test', 
         output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_huberts_and_freezed_vits_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')

    # write MBR decode outputs to file for force_aligned_whole_hubert*_embed512_lr5e-6_83k-5k-5k
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
         data_split = 'val', 
         output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_huberts_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr-self_train', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
         data_split = 'train', 
         output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_huberts_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr'], 
         data_split = 'test', 
         output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_huberts_embed512_lr5e-6_83k-5k-5k.txt')
    
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'val',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/val/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr-self_train'], 
        data_split = 'train',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/train/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k.txt')

    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test', 
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k.txt')
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr'], 
        data_split = 'test',
        output_fpath = 'exp/spokencoco/MBR_decode_trees/test/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k.txt')

    ######################################################################## MBR selection for phn_MFA diffBounad whole_hubert #############################################################################
    # run MBR for phn MFA diffboundary V0
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    
    run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])

    # run MBR for phn MFA diffboundary V1
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])

    run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
        'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])

    ########################################################################### MBR selection for unsup-discovery seg_feats ##################################################################################
    # MBR across epoches, learning rates, mlp_combine but within the same unsup-discovery seg_feats
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')

    # MBR across epoches and learning rates but within the same unsup-discovery seg_feats
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')

    # MBR across epoches but within the same learning rate and within the same unsup-discovery seg_feats
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV3_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombineV2_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_MLPcombine_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.7_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-3_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-4_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-5_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    run(['exp/spokencoco/unsup_attn_discovery_disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest_seg_feats_embed512_lr1e-6_83k-5k-5k/mbr'], 
        unsup_word_discovery_feats = 'disc-82_spokencoco_preFeats_weightedmean_0.8_9_clsAttn_vadpython_insertThreshold0.2_snapshotbest', 
        unsup_word_discovery_feat_type = 'attn')
    
    ########################################################################### MBR selection for MFA whole_hubert ##################################################################################
    # MBR across epoches, learning rates, and hubert layers
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-3_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-4_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr'])
    
    # MBR across epoches and learning rates but the same hubert layers
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-5_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr'])
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
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr1e-5_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-3_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-4_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr5e-6_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr1e-5_83k-5k-5k/mbr', 
        'exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr1e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
        'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])

    # MBR across epoches and hubert layers but the same learning rate
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr1e-5_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr1e-5_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr1e-5_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr1e-5_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr', 
         'exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr',
         'exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])

    # MBR across epoches but within a hyper-parameter set 
    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb16_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits16_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vitb8_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_vits8_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_patch16_224_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr1e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-7_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_freezed_deit_base_distilled_patch16_384_whole_hubert2_embed512_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr5e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr1e-5_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed768_lr1e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-3_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-4_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-5_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr5e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert_large24_embed512_lr1e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert2_embed512_lr5e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert4_embed512_lr5e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert6_embed512_lr5e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert8_embed512_lr5e-6_83k-5k-5k/mbr']) 
    run(['exp/spokencoco/force_aligned_whole_hubert10_embed512_lr5e-6_83k-5k-5k/mbr'])
    run(['exp/spokencoco/force_aligned_whole_hubert_embed512_lr5e-6_83k-5k-5k/mbr']) 
