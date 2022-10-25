import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import os

from module import AttentivePooling
from utils import l2norm

################################## get word boundary metric ##################################
import numpy as np
def find_boundary_matches(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
        e.g. [0.07, 0.12, 0.33, 0.45, 1.10]
    pred: list of predicted boundaries
        e.g. same format as gt
    tolerance: 0.02
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    while gt_pointer < gt_len and pred_pointer < pred_len:
        if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
            match_gt += 1
            match_pred += 1
            gt_pointer += 1
            pred_pointer += 1
        elif gt[gt_pointer] > pred[pred_pointer]:
            pred_pointer += 1
        else:
            gt_pointer += 1
    # # below is method used in Kreuk et al, which I didn't used for SpokenCOCO
    # for pred_i in pred:
    #     min_dist = np.abs(gt - pred_i).min()
    #     match_pred += (min_dist <= tolerance)
    # for y_i in gt:
    #     min_dist = np.abs(pred - y_i).min()
    #     match_gt += (min_dist <= tolerance)
    return match_gt, match_pred, gt_len, pred_len

def run_word_seg_F1(all_pred_boundaries, all_gt_boundaries): 
    match_gt_count = 0
    match_pred_count = 0
    gt_b_len = 0
    pred_b_len = 0
    tolerance = 0.02 # in seconds, this requires that boundaries are also in seconds
    for pred_boundaries, gt_boundaries in zip(all_pred_boundaries, all_gt_boundaries): # loop through SpokenCOCO dataset
        a, b, c, d = find_boundary_matches(gt_boundaries[1:-1], pred_boundaries[1:-1], tolerance) # exclude the first and last boundaries, you don't have to, but I did this in the paper
        match_gt_count += a
        match_pred_count += b
        gt_b_len += c
        pred_b_len += d

    b_prec = match_pred_count / pred_b_len
    b_recall = match_gt_count / gt_b_len
    b_f1 = 2*b_prec*b_recall / (b_prec+b_recall)
    b_os = b_recall / b_prec - 1.
    b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
    b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
    b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
    print("precision: ", b_prec)
    print("recall: ", b_recall)
    print("f1: ", b_f1)
    print("oversegmentation: ", b_os)
    print("R-value: ", b_r_val)

    return b_f1

def generate_predict_word_seg(phn_to_word_map, phn_seg): 
    # generate predicted word segmentation based on 
    # (1) an existing phone segmentation 
    # (2) a phone-to-word mapping 
    
    phn_to_word_map = [-1] + phn_to_word_map
    predict_word_switch_indices = [(i-1) for i in range(1, len(phn_to_word_map)) if (phn_to_word_map[i] != 0 and phn_to_word_map[i] != phn_to_word_map[i-1])]

    predicted_word_list = []
    for i in range(0, len(predict_word_switch_indices)-1): 
        curr_start = predict_word_switch_indices[i]
        curr_end = predict_word_switch_indices[i+1] 

        predicted_phn_span = phn_seg[curr_start: curr_end]
        predicted_curr_word = ('shit', predicted_phn_span[0][1], predicted_phn_span[-1][2])
        predicted_word_list.append(predicted_curr_word)
    # append the last phn 
    predicted_phn_span = phn_seg[curr_end:]
    predicted_curr_word = ('shit', predicted_phn_span[0][1], predicted_phn_span[-1][2])
    predicted_word_list.append(predicted_curr_word)
       
    return predicted_word_list

def convert_word_list_format(word_list): 
    # convert word list to the following format: 
    # [0.07, 0.12, 0.33, 0.45, 1.10] (in seconds)
    
    reformat_word_list = [word_list[0][1]]
    prev_time_stamp = word_list[0][2]
    for i in range(1, len(word_list)):
        curr_time_stamp = word_list[i][1]
        curr_time_stamp = (curr_time_stamp + prev_time_stamp) / 2
        reformat_word_list.append(curr_time_stamp)
        prev_time_stamp = word_list[i][2]
    reformat_word_list.append(word_list[-1][-1])

    return [x/100 for x in reformat_word_list]

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = 'data/SpokenCOCO/Freda-formatting/')
    args = parser.parse_args()

    # load oracle word, phone seg
    oracle_phn_seg = np.load(os.path.join(args.data_path, 'test_segment-hubert2_phn_list-83k-5k-5k.npy'), allow_pickle=True)[0]
    oracle_word_seg = np.load(os.path.join(args.data_path, 'test_segment-hubert2_word_list-83k-5k-5k.npy'), allow_pickle=True)[0]


    best_bf1 = 0
    for ckpt in range(0, 10): 
        # load predicted phn-to-word mapping
        b = np.load(f'exp/spokencoco/phn_force_aligned_diffboundV3-gtword_whole_hubert2_embed512_lr5e-6_margin0.2_lambdahi0_83k-5k-5k/predict_word_seg/{ckpt}_pred_word_seg.npy', allow_pickle=True)

        # calculate word segmentation F1
        all_pred_boundaries, all_gt_boundaries = [], []
        for idx in range(len(b)): 
            predicted_word_seg = generate_predict_word_seg(b[idx], oracle_phn_seg[idx])
            assert len(predicted_word_seg) == len(oracle_word_seg[idx])

            predicted_word_seg_reformat = convert_word_list_format(predicted_word_seg)
            oracle_word_seg_reformat = convert_word_list_format(oracle_word_seg[idx])
            #print(predicted_word_seg_reformat)
            #print(oracle_word_seg_reformat)

            all_pred_boundaries.append(predicted_word_seg_reformat)
            all_gt_boundaries.append(oracle_word_seg_reformat)

        b_f1 = run_word_seg_F1(all_pred_boundaries, all_gt_boundaries)
        best_bf1 = max(best_bf1, b_f1)

    print('best is', best_bf1)

