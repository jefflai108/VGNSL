import collections
import itertools
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import sys

from evaluation import _retrieve_text_from_tree, _cleanup_tree, extract_statistics, extract_spans

sys.path.insert(-1, os.path.join(sys.path[0], '../analysis'))
from ex_sparseval import corpus_f1 as ex_sparseval_f1

sys.path.insert(-1, os.path.join(sys.path[0], '/data/sls/scratch/clai24/syntax/AutowordSE/src'))
from autowordse import BareTree, ExtendedTree, TreeAligner

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

def run(pred_tree_dir_list, unsup_word_discovery_feats=None, unsup_word_discovery_feat_type=None, data_split='test'): 
    # standard files
    ground_truth_trees = _open_tree_file(f'data/SpokenCOCO/Freda-formatting/{data_split}_word-level-ground-truth-83k-5k-5k.txt')
    oracle_bounds = np.load(f'data/SpokenCOCO/Freda-formatting/{data_split}_segment-hubert2_word_list-83k-5k-5k.npy', allow_pickle=True)[0]
    for i, oracle_bound in oracle_bounds.items(): 
        oracle_bound_in_time = [(word, start_time * 0.02, end_time * 0.02) for (word, start_time, end_time) in oracle_bound]
        oracle_bounds[i] = oracle_bound_in_time
    if unsup_word_discovery_feats:
        word_bounds = np.load(f'data/SpokenCOCO/Freda-formatting/{data_split}-{unsup_word_discovery_feats}-pred_{unsup_word_discovery_feat_type}_list-83k-5k-5k.npy', allow_pickle=True)[0]
    else: 
        word_bounds = oracle_bounds

    # open all ckpt's pred_trees
    pred_tree_files = []
    for pred_tree_dir in pred_tree_dir_list:
        pred_tree_files.extend([osp.join(pred_tree_dir, f) for f in os.listdir(pred_tree_dir) if osp.isfile(osp.join(pred_tree_dir, f))])
    pred_trees = [_open_tree_file(pred_tree_file) for pred_tree_file in pred_tree_files]

    # mbr selection across ckpts
    predicted_tree_accum_iou, rbt_accum_iou = 0, 0
    cnt = 0
    for i in tqdm(range(len(ground_truth_trees))): 
        if ground_truth_trees[i] == 'MISMATCH': 
            continue 
        cnt += 1

        if len(word_bounds[i]) >= 2:
            pred_tree_samples = [pred_tree[i] for pred_tree in pred_trees] # outputs from different model/ckpt given an input
            output = mbr_selection(pred_tree_samples, key_function=pairwise_f1_score_for_mbr) # agreement score based on normal F1
            def pairwise_autowordse_score_for_mbr(induced_tree_1, induced_tree_2):
                return sample_wise_autowordse(induced_tree_1, induced_tree_2, word_bounds[i], word_bounds[i])

            #output = mbr_selection(pred_tree_samples, key_function=pairwise_autowordse_score_for_mbr) # agreement score based on autowordse 
            mbr_selected_tree = output['best_sample']

            predicted_avg_iou, rbt_avg_iou = sample_wise_autowordse(ground_truth_trees[i], mbr_selected_tree, oracle_bounds[i], word_bounds[i])
        else: # give 0% to IOUs
            #print(idx)
            #print(gold_trees[idx])
            #print(induced_trees[idx])
            #print(oracle_bounds[idx])
            #print(word_bounds[idx])
            predicted_avg_iou = 0 
            rbt_avg_iou = 0

        predicted_tree_accum_iou += predicted_avg_iou
        rbt_accum_iou += rbt_avg_iou

    predicted_avg_iou = predicted_tree_accum_iou / cnt 
    rbt_avg_iou = rbt_accum_iou / cnt 
    print(f'Predicted Tree avg IOU is {predicted_avg_iou:.3f} and RBT avg IOU is {rbt_avg_iou:.3f}')

def sample_wise_autowordse(gold_tree, induced_tree, oracle_bound, word_bound): 
    datum = {
        'gold_tree': BareTree.fromstring(gold_tree),
        'induced_tree': BareTree.fromstring(induced_tree),
        'oracle_bounds': oracle_bound,
        'word_bounds': word_bound
    }
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

    return predicted_avg_iou, rbt_avg_iou

if __name__ == '__main__':
    
    stage = int(sys.argv[1])
    print(stage) 

    ########################################################## MBR selection for MBR_unsup-discovery MBR_seg_feats (fully-unsup setting) ##################################################################
    # MBR across epoches and hubert layers but the same learning rate
    if stage == 0: 
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer9_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr',
            'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer10_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr', 
            'exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer11_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'],
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')

    elif stage == 1:
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
            unsup_word_discovery_feat_type = 'word')

    elif stage == 2:
        # MBR across epoches but within a hyper-parameter set
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer0_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 3:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer1_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 4:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer2_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 5:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer3_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 6:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer4_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 7:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer5_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 8:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer6_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 9:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer7_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 10:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer8_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 11:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer9_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 12:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer10_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')
    elif stage == 13:
        run(['exp/spokencoco/mbr_unsup_attn_discovery_mbr_104_1030_top10_mbr_seg_feats_disc-81_snapshot15_layer11_embed512_MLPcombineV3_lr1e-3_83k-5k-5k/mbr'], 
            unsup_word_discovery_feats = 'mbr_104_1030_top10', 
            unsup_word_discovery_feat_type = 'word')

    ######################################################################## MBR selection for phn_MFA diffBounad whole_hubert #############################################################################
    # run MBR for phn MFA diffboundary V0
    elif stage == 14:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr']) 
    elif stage == 15:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 16:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 17:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 18:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 19:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
        
    elif stage == 20:
        run(['exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV0-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])

        # run MBR for phn MFA diffboundary V1
    elif stage == 21:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 22:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 23:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 24:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 25:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    elif stage == 26:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    
    elif stage == 27:
        run(['exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert4_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert6_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert8_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert10_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr',
            'exp/spokencoco/phn_force_aligned_diffboundV1-gtword_whole_hubert_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/mbr'])
    exit()

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
