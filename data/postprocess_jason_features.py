import os.path as osp
import json 
import pickle
from tqdm import tqdm
import argparse

import torch
import numpy as np 
import textgrid

from networkx_tutorial import run as l1_alignment

from utils import vghubert_feature_extraction, setup_vg_hubert

def _convert_default_wav_to_wavspeaker(main_dir, wav_pth): 
    wav_name = wav_pth.split('/')[-1]
    spk = wav_pth.split('/')[-1].split('-')[0]
    return osp.join(main_dir, 'wavs-speaker', spk, wav_name)

def run(data_summary_json, wav2wordseg, target_word_list_pth, target_alignment_pth, weighting_via_l1=True):
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
                gt_word_list = read_textgrid(alignment_file, transcript_file, pred_word_segment[-1][-1].item())
                pred_word_list_dict[idx] = pred_word_list
            else: 
                pred_word_list = pred_word_lists[idx]
                gt_word_list = read_textgrid(alignment_file, transcript_file, pred_word_list[-1][-1].item())
                pred_word_segment = np.array([[x[1], x[2]] for x in pred_word_list])
            #print(pred_word_list) # store this in the same format as gt_word_list
            #print(gt_word_list)

            alignment = l1_alignment(gt_word_list, pred_word_segment, weighting_via_l1=weighting_via_l1)
            #print(alignment) # store this in the same order as others
            alignment_dict[idx] = alignment

            idx += 1

    if wav2wordseg:
        np.save(target_word_list_pth, [pred_word_list_dict])
    np.save(target_alignment_pth, [alignment_dict])

def run_seg_feat(data_summary_json, wav2wordseg, target_seg_feat_pth):
    seg_feat_dict = {}
    idx = 0
    for image_key, captions_list in tqdm(data_summary_json.items()):
        captions_list = _deduplicate(captions_list, image_key)
        for caption in captions_list: 
            wav_pth = caption[0]
            pred_seg_feat = wav2wordseg[wav_pth]
            seg_feat_dict[idx] = pred_seg_feat

            idx += 1

    np.save(target_seg_feat_pth, [seg_feat_dict])

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

def read_jason_file(main_dir, ffile, boundaries_only=True, attention_boundaries=True, upstream_model=None, upstream_model_layer=11, device='cuda'): 
    with open(ffile, 'rb') as f: 
        features = pickle.load(f)

    attn_boundaries = {_convert_default_wav_to_wavspeaker(main_dir, k):v['boundaries'] for k,v in features.items()}
    if not attention_boundaries: 
        word_boundaries = {_convert_default_wav_to_wavspeaker(main_dir, k):torch.tensor(v['word_boundaries']) for k,v in features.items()}
    print(len(features.keys())) # 25020 for val, 567171 for train, 25031 for test

    if boundaries_only: # return attn/word boundaries 
        if attention_boundaries: 
            return attn_boundaries
        else: return word_boundaries
    else: # return seg_feats instead of bounadries
        if upstream_model: # seg_features do not exist --> extract and pre-store 
            seg_features = {} 
            for wav_file in features.keys(): 
                wav_file_fixed = _convert_default_wav_to_wavspeaker(main_dir, wav_file)
                attn_boundaries = features[wav_file]['boundaries'] # load attention segment boundaries for [CLS]-weighted mean-pool 
                vghubert_repre, _ = vghubert_feature_extraction(wav_file_fixed, 
                                                                upstream_model, 
                                                                layer=upstream_model_layer, 
                                                                device=device, 
                                                                cls_mean_pool=True, 
                                                                spf=features[wav_file]['spf'],
                                                                boundaries=attn_boundaries)
                seg_feature = np.transpose(vghubert_repre)
                assert seg_feature.shape[0] == len(attn_boundaries)
                #print(seg_feature.shape) # (13, 768) 
                seg_features[wav_file_fixed] = seg_feature
        else: # seg_features exist --> pre-store only 
            seg_features = {_convert_default_wav_to_wavspeaker(main_dir, k):v['seg_feats'] for k,v in features.items()}
            for wav in attn_boundaries.keys(): 
                boundary = attn_boundaries[wav]
                seg_feature = seg_features[wav]
                assert len(boundary) == len(seg_feature) # ensure 1:1 mapping 

        return seg_features

def convert_attention_boundary_to_word_boundary(attn_list_file, word_list_file):
	# convert Jason's attention boundaries to word boundaries 
	# input should be in the same units (in seconds)

    all_attn_boundaries = np.load(attn_list_file, allow_pickle=True)[0]
    all_word_boundaries = {}
    for k, attn_boundaries in all_attn_boundaries.items():
        attn_boundaries = [[x[1], x[2]] for x in attn_boundaries]

        # determine last frame 
        pred_last_frame = attn_boundaries[-1][-1]
        final_last_frame = pred_last_frame

        # locate word boundaries via attention endpoints avg
        temp_word_boundaries = [(0 + attn_boundaries[0][0])/2]
        for left, right in zip(attn_boundaries[:-1], attn_boundaries[1:]):
            temp_word_boundaries.append((left[1] + right[0])/2)
        temp_word_boundaries.append(final_last_frame)

        # generate final word boundaries 
        word_boundaries = []
        for left, right in zip(temp_word_boundaries[:-1], temp_word_boundaries[1:]):
            word_boundaries.append([left, right])
        word_boundaries = [('shit', x[0], x[1]) for x in word_boundaries]

        # store word boundaries 
        all_word_boundaries[k] = word_boundaries
    
    np.save(word_list_file, [all_word_boundaries])

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_summary_json', '-j', 
                        type=str, default='data/SpokenCOCO/SpokenCOCO_summary-83k-5k-5k.json')
    parser.add_argument('--data_dir', type=str, default='data/SpokenCOCO')
    parser.add_argument('--data_split', type=str, choices=['test', 'val', 'train'])
    parser.add_argument('--vghubert_layer', type=int, choices=[0,1,2,3,4,5,6,7,8,9,10,11])
    parser.add_argument('--new_feature', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()

    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json) as f: 
        coco_json = json.load(f)

    # Step 1: attention boundaries & alignment based on attention boundaries
    attn_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_attn_list-' + basename + '.npy')
    attn_alignment_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-attn_alignment_via_max_weight_matching-' + basename + '.npy')
    print('storing pred_attn_list at %s\nstoring alignment at %s\n' % (attn_list_file, attn_alignment_file))
    wav2wordseg = read_jason_file(args.data_dir, 
                                  osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'))
    run(coco_json[args.data_split], wav2wordseg, attn_list_file, attn_alignment_file)

    # (Optional) Step 2.1: word boundaries & alignment based on word boundaries
    word_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_word_list-' + basename + '.npy')
    word_alignment_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-word_alignment_via_max_weight_matching-' + basename + '.npy')
    print('storing pred_word_list at %s\nstoring alignment at %s\n' % (word_list_file, word_alignment_file))
    wav2wordseg = read_jason_file(args.data_dir, 
                                  osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'), 
                                  attention_boundaries=False)
    run(coco_json[args.data_split], wav2wordseg, word_list_file, word_alignment_file)

    # (Optional) Step 2.2.1: convert attention boundaries to word boundaries
    attn_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_attn_list-' + basename + '.npy')
    word_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_word_list-' + basename + '.npy')
    print('converting pred_attn_list %s\nto pred_word_list %s\n' % (attn_list_file, word_list_file))
    convert_attention_boundary_to_word_boundary(attn_list_file, word_list_file)
   
    # (Optional) Step 2.2.2: generate alignment based on word boundaries 
    word_alignment_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-word_alignment_via_max_weight_matching-' + basename + '.npy')
    print('storing pred_word_list at %s\nstoring alignment at %s\n' % (word_list_file, word_alignment_file))
    run(coco_json[args.data_split], None, word_list_file, word_alignment_file)

    # (Optional) Step 2.3: generete additional (duration-based) alignments based on existing attn_list/word_list
    attn_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_attn_list-' + basename + '.npy')
    assert osp.exists(attn_list_file)
    attn_alignment_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-attn_alignment_via_max_weight_duration_matching-' + basename + '.npy')
    print('storing pred_attn_list at %s\nstoring alignment at %s\n' % (attn_list_file, attn_alignment_file))
    run(coco_json[args.data_split], None, attn_list_file, attn_alignment_file, weighting_via_l1=False)

    word_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_word_list-' + basename + '.npy')
    assert osp.exists(word_list_file)
    word_alignment_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-word_alignment_via_max_weight_duration_matching-' + basename + '.npy')
    print('storing pred_word_list at %s\nstoring alignment at %s\n' % (word_list_file, word_alignment_file))
    run(coco_json[args.data_split], None, word_list_file, word_alignment_file, weighting_via_l1=False)
    
    # Step 3: pre-store Jason's default feature
    seg_feat_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_seg_feat-' + basename + '.npy')
    wav2wordseg = read_jason_file(args.data_dir, 
                                  osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'), 
                                  boundaries_only=False)
    print('storing pred_seg_feat at %s' % seg_feat_file)
    run_seg_feat(coco_json[args.data_split], wav2wordseg, seg_feat_file)

    # (Optional) Step 3.1: extract and pre-store Jason's default feature
    seg_feat_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-disc-81_snapshot15_layer${args.vghubert_layer}-pred_seg_feat-' + basename + '.npy')
    device = 'cuda'
    upstream_model = setup_vg_hubert(model_type='disc-81', snapshot='15', device=device) # load disc-81, snapshot 15
    wav2wordseg = read_jason_file(args.data_dir, 
                                  osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'), 
                                  boundaries_only=False, 
                                  upstream_model=upstream_model, 
                                  upstream_model_layer=args.vghubert_layer, 
                                  device=device)
    print('storing pred_seg_feat at %s' % seg_feat_file)
    run_seg_feat(coco_json[args.data_split], wav2wordseg, seg_feat_file)
