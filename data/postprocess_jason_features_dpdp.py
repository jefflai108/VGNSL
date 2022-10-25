import os.path as osp
import json 
import pickle
from tqdm import tqdm
import argparse
import s3prl.hub as hub
import h5py

import torch
import numpy as np 
import textgrid

from networkx_tutorial import run as l1_alignment

from utils import vghubert_feature_extraction, setup_vg_hubert, hubert_feature_extraction_mp

def _convert_default_wav_to_wavspeaker(main_dir, wav_pth): 
    wav_name = wav_pth.split('/')[-1]
    wav_name = wav_name + '.wav'
    spk = wav_pth.split('/')[-1].split('-')[0]
    return osp.join(main_dir, 'wavs-speaker', spk, wav_name)

def run(data_summary_json, wav2wordseg, target_word_list_pth):
    if wav2wordseg is None: 
        pred_word_lists = np.load(target_word_list_pth, allow_pickle=True)[0]
    
    pred_word_list_dict = {}
    alignment_dict = {}
    idx = 0
    for image_key, captions_list in tqdm(data_summary_json.items()):
        captions_list = _deduplicate(captions_list, image_key)
        for caption in captions_list: 
            wav_pth = caption[0]
            transcript_file = caption[1]
            alignment_file = caption[3]
        
            if wav2wordseg:
                pred_word_segment = wav2wordseg[wav_pth]
                pred_word_list = _generate_pretty_word_list(pred_word_segment)
                pred_word_list_dict[idx] = pred_word_list
            else: 
                pred_word_list = pred_word_lists[idx]
                gt_word_list = read_textgrid(alignment_file, transcript_file, pred_word_list[-1][-1].item())
                pred_word_segment = np.array([[x[1], x[2]] for x in pred_word_list])
            #print(pred_word_list) # store this in the same format as gt_word_list
            #print(gt_word_list)

            idx += 1

    if wav2wordseg:
        np.save(target_word_list_pth, [pred_word_list_dict])


def _generate_pretty_word_list(pred_word_segment): 
    # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
    # since Jason's word segments do not correspond to any particular words, we use a dummay 
    # word token as replacement, and in this case "shit"
    return [('shit', x[0], x[1]) for x in np.array(pred_word_segment)]
    

def read_textgrid(textgrid_pth, text_pth, pred_last_timestamp, epsilon=0.05):
    # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
    word_tgs = textgrid.TextGrid.fromFile(textgrid_pth)[0]

    word_list = []
    for word_tg in word_tgs: 
        word = word_tg.mark
        if word == '': # probably silence
            continue 
        word_obj = (word, word_tg.minTime, word_tg.maxTime)
        word_list.append(word_obj)
    assert word_tgs[-1].maxTime + epsilon >= pred_last_timestamp, print(word_tgs[-1], pred_last_timestamp)

    return word_list
 
def _deduplicate(captions_list, image_key):
    # ensure image:captions == 1:5
    if len(captions_list) > 5: 
        captions_list = captions_list[:5]
    while len(captions_list) < 5: # duplicate 
        print('duplicate %s captions' % image_key)
        captions_list.append(captions_list[-1])
    assert len(captions_list) == 5

    return captions_list

def read_jason_file(main_dir, ffile, boundaries_only=True, pre_extracted_hubert_repres=None): 
    with open(ffile, 'rb') as f: 
        features = pickle.load(f)

    boundaries = {_convert_default_wav_to_wavspeaker(main_dir, k):v for k,v in features.items()}
    print(len(features.keys())) # 25020 for val, 567171 for train, 25031 for test

    if boundaries_only: # return attn/word boundaries 
        return boundaries
    else: # return seg_feats instead of bounadries
        if pre_extracted_hubert_repre:
            seg_features = {} 
            idx = 0
            for wav_file in tqdm(features.keys()): 
                wav_file_fixed = _convert_default_wav_to_wavspeaker(main_dir, wav_file)
                boundaries = features[wav_file] # load attention segment boundaries for [CLS]-weighted mean-pool 
                seg_feature, _ = hubert_feature_extraction_mp(wav_file_fixed, 
                                                              pre_extracted_hubert_repre[str(idx)][:], 
                                                              spf=0.02,
                                                              boundaries=boundaries)
                assert seg_feature.shape[0] == len(boundaries)
                #print(seg_feature.shape) # (13, 768) 
                seg_features[wav_file_fixed] = seg_feature
                #print(wav_file_fixed)
                idx += 1

        return seg_features

def run_seg_feat(data_summary_json, target_seg_feat_pth, main_dir, ffile, pre_extracted_hubert_repres=None):
    with open(ffile, 'rb') as f: 
        features = pickle.load(f)

    boundaries = {_convert_default_wav_to_wavspeaker(main_dir, k):v for k,v in features.items()}
    print(len(features.keys())) # 25020 for val, 567171 for train, 25031 for test

    seg_feat_dict = {}
    idx = 0
    for image_key, captions_list in tqdm(data_summary_json.items()):
        captions_list = _deduplicate(captions_list, image_key)
        for caption in captions_list: 
            wav_pth = caption[0]
            boundaries = features[wav_pth.split('/')[-1][:-4]]
            whole_hubert_repre = pre_extracted_hubert_repres[str(idx)][:]

            # double-check 
            if len(boundaries) >= 1: 
                last_boundary = boundaries[-1][-1] / 0.02
                assert last_boundary <= len(whole_hubert_repre)

            seg_feature, _ = hubert_feature_extraction_mp(wav_pth, 
                                                          whole_hubert_repre, 
                                                          spf=0.02,
                                                          boundaries=boundaries)
            assert seg_feature.shape[0] == len(boundaries)
            #print(seg_feature.shape) # (13, 768) 
            seg_feat_dict[idx] = seg_feature

            idx += 1

    np.save(target_seg_feat_pth, [seg_feat_dict])

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_summary_json', '-j', 
                        type=str, default='data/SpokenCOCO/SpokenCOCO_summary-83k-5k-5k.json')
    parser.add_argument('--data_dir', type=str, default='data/SpokenCOCO')
    parser.add_argument('--data_split', type=str, choices=['test', 'val', 'train'])
    parser.add_argument('--vghubert_layer', type=int, choices=[1,2,3,4,5,6,7,8,9,10,11,12])
    parser.add_argument('--new_feature', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()

    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json) as f: 
        coco_json = json.load(f)

    ## Step 1: store boundaries
    #attn_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_attn_list-' + basename + '.npy')
    #print('storing pred_attn_list at %s\n' % (attn_list_file))
    #wav2wordseg = read_jason_file(args.data_dir, 
    #                              osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'))
    #run(coco_json[args.data_split], wav2wordseg, attn_list_file)

    # Step 2: extract and pre-store Jason's default feature
    seg_feat_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-hubert{args.vghubert_layer}-pred_seg_feat-' + basename + '.npy')
    pre_extracted_hubert_repre = h5py.File(osp.join(args.output_dir, f'{args.data_split}_segment-hubert{args.vghubert_layer}_embed-' + basename + '.hdf5'), 'r')

    print('storing pred_seg_feat at %s' % seg_feat_file)
    run_seg_feat(coco_json[args.data_split], 
                 seg_feat_file, 
                 args.data_dir, 
                 osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'), 
                 pre_extracted_hubert_repres=pre_extracted_hubert_repre)
