import os.path as osp
import json 
import pickle
from tqdm import tqdm
import argparse

import numpy as np 
import textgrid

from networkx_tutorial import run as l1_alignment

# 1. map "wav" --> location in "wavs-speaker"

# e.g. "wavs/train/428/mfbjxyldr6c1o-3ZPPDN2SLVWRQHXHKC1P7P9QLHI9EC_296759_180764.wav" 
# shoudl be "wavs-speaker/mfbjxyldr6c1o/mfbjxyldr6c1o-3ZPPDN2SLVWRQHXHKC1P7P9QLHI9EC_296759_180764.wav"
#
# 2. SpokenCOCO summary: SpokenCOCO_summary-83k-5k-5k.json
# find the corresponding files 
#
#
# 2.1 make sure the time stamps align (frame rate)
#
# 3. modify word_list (create a new one) with new word boundaries
# e.g. y = np.load('test_segment-hubert2_word_list-83k-5k-5k.npy', allow_pickle=True)
# >>> y[0][0]
#[('a', 33.5, 41.5), ('dirt', 41.5, 61.0), ('path', 61.0, 86.5), ('with', 86.5, 92.5), ('a', 92.5, 96.0), ('young', 96.0, 112.00000000000001), ('person', 112.00000000000001, 132.5), ('on', 132.5, 140.0), ('a', 140.0, 144.5), ('motor', 144.5, 159.5), ('bike', 159.5, 178.0), ('rests', 183.5, 214.5), ('to', 216.0, 221.49999999999997), ('the', 221.49999999999997, 226.5), ('foreground', 226.5, 260.0), ('of', 260.0, 265.0), ('a', 265.0, 268.5), ('verdant', 268.5, 292.0), ('area', 293.5, 314.5), ('with', 314.5, 324.0), ('a', 324.0, 327.5), ('bridge', 327.5, 357.0), ('and', 366.5, 373.0), ('a', 373.0, 377.5), ('background', 377.5, 410.50000000000006), ('of', 410.50000000000006, 416.5), ('cloudwreathed', 423.00000000000006, 469.00000000000006), ('mountains', 476.99999999999994, 509.49999999999994)]
##

# goal of the above: 
# first step: construct a dictionary {new_wav_id: y'[0][0]} 
# second step: min-weight matching of y[0][0] and y'[0][0]
# third step: output in Freda format by running same procedure of data/reformat_v4.py 

def _convert_default_wav_to_wavspeaker(main_dir, wav_pth): 
    wav_name = wav_pth.split('/')[-1]
    spk = wav_pth.split('/')[-1].split('-')[0]
    return osp.join(main_dir, 'wavs-speaker', spk, wav_name)

def run(data_summary_json, wav2wordseg, target_word_list_pth, target_alignment_pth):
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

            alignment = l1_alignment(gt_word_list, pred_word_segment)
            #print(alignment) # store this in the same order as others
            alignment_dict[idx] = alignment

            idx += 1

    if wav2wordseg:
        np.save(target_word_list_pth, [pred_word_list_dict])
    np.save(target_alignment_pth, [alignment_dict])

def _generate_pretty_word_list(pred_word_segment): 
    # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
    # since Jason's word segments do not correspond to any particular words, we use a dummay 
    # word token as replacement, and in this case "shit"
    return [('shit', x[0], x[1]) for x in np.array(pred_word_segment)]
    

def read_textgrid(textgrid_pth, text_pth, pred_last_timestamp, epsilon=0.001):
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

def read_jason_file(main_dir, ffile): 
    with open(ffile, 'rb') as f: 
        features = pickle.load(f)

    features = {_convert_default_wav_to_wavspeaker(main_dir, k):v['boundaries'] for k,v in features.items()}
    print(len(features.keys())) # 25020 for val, 567171 for train, 25031 for test
    return features

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
    wav2wordseg = read_jason_file(args.data_dir, osp.join(args.data_dir, 'Jason_word_discovery', args.new_feature, args.data_split + '_data_dict.pkl'))
    run(coco_json[args.data_split], wav2wordseg, attn_list_file, attn_alignment_file)

    # Step 2: convert attention boundaries to word boundaries
    attn_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_attn_list-' + basename + '.npy')
    word_list_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-pred_word_list-' + basename + '.npy')
    print('converting pred_attn_list %s\nto pred_word_list %s\n' % (attn_list_file, word_list_file))
    convert_attention_boundary_to_word_boundary(attn_list_file, word_list_file)
   
    # Step 3: generate alignment based on word boundaries 
    word_alignment_file = osp.join(args.output_dir, f'{args.data_split}-{args.new_feature}-word_alignment_via_max_weight_matching-' + basename + '.npy')
    print('storing pred_word_list at %s\nstoring alignment at %s\n' % (word_list_file, word_alignment_file))
    run(coco_json[args.data_split], None, word_list_file, word_alignment_file)

